const wekua = @import("../../wekua.zig");
const std = @import("std");
const cl = @import("opencl");

const w_activation = wekua.nn.activation;
const w_layer = @import("main.zig");

const core = wekua.core;
const KernelsSet = core.KernelsSet;
const CommandQueue = core.CommandQueue;

const Tensor = wekua.Tensor;

const bias_cl_kernel: []const u8 = @embedFile("kernels/bias.cl");
const bias_step_cl_kernel: []const u8 = @embedFile("kernels/bias_step.cl");

pub const LinearExtraParams = struct {
    deep: usize = 1,
    enable_bias: bool = true,
    use_complex_numbers: bool = false,
};

inline fn get_random_limits(comptime T: type, input: usize, output: usize, start: *T, end: *T) void {
    const limit = switch (@typeInfo(T)) {
        .int => std.math.sqrt(6 / (input + output)),
        .float => std.math.sqrt(6.0 / @as(T, @floatFromInt(input + output))),
        else => unreachable,
    };

    start.* = switch (@typeInfo(T)) {
        .int => |info| blk: {
            if (info.signedness == .signed) {
                break :blk -limit;
            }
            break :blk 0;
        },
        else => -limit,
    };
    end.* = limit;
}

pub fn Linear(
    comptime T: type,
) type {
    const LayerTensor = Tensor(T);
    const Layer = w_layer.Layer(T);

    const LinearCache = struct {
        outputs: []*LayerTensor,
        sensitivities: []*LayerTensor,
        acti_derivatives: []*LayerTensor,

        gradients: []*LayerTensor,
        bias_gradients: []*LayerTensor,
        bias_gradients_lis: []u64,
    };

    return struct {
        allocator: std.mem.Allocator,

        weights: []*LayerTensor,

        bias_enabled: bool,
        bias: []?*LayerTensor,

        activation: ?w_activation.Activation(T),

        context: *const wekua.core.Context,

        const Self = @This();

        pub fn init(
            context: *const wekua.core.Context,
            input: usize,
            output: usize,
            activation: ?w_activation.Activation(T),
            extra_params: LinearExtraParams,
        ) !Layer {
            if (input == 0 or output == 0 or extra_params.deep == 0) return wekua.tensor.Errors.InvalidValue;

            const allocator = context.allocator;

            const layer = try allocator.create(Self);
            errdefer allocator.destroy(layer);

            layer.activation = activation;
            layer.allocator = allocator;
            layer.context = context;
            layer.bias_enabled = extra_params.enable_bias;

            var bottom: T = undefined;
            var top: T = undefined;
            get_random_limits(T, input, output, &bottom, &top);

            const weights = try allocator.alloc(*LayerTensor, extra_params.deep);
            layer.weights = weights;

            var weights_created: usize = 0;
            errdefer {
                for (weights[0..weights_created]) |w| {
                    w.release();
                }
                allocator.free(weights);
            }

            const first_weight_layer = try LayerTensor.alloc(context, &.{ output, input }, .{
                .is_complex = extra_params.use_complex_numbers,
            });
            weights[0] = first_weight_layer;
            weights_created += 1;

            const default_command_queue = &context.command_queues[0];
            try wekua.tensor.random.uniform(
                T,
                default_command_queue,
                first_weight_layer,
                null,
                bottom,
                top,
            );

            get_random_limits(T, output, output, &bottom, &top);
            for (weights[1..]) |*w| {
                const weight = try LayerTensor.alloc(context, &.{ output, output }, .{
                    .is_complex = extra_params.use_complex_numbers,
                });
                errdefer weight.release();

                try wekua.tensor.random.uniform(T, default_command_queue, weight, null, bottom, top);

                w.* = weight;
                weights_created += 1;
            }

            if (extra_params.enable_bias) {
                const bias_layers = try allocator.alloc(?*LayerTensor, extra_params.deep);
                layer.bias = bias_layers;

                var bias_created: usize = 0;
                errdefer {
                    for (bias_layers[0..bias_created]) |b| {
                        b.?.release();
                    }
                    allocator.free(bias_layers);
                }

                for (bias_layers) |*b| {
                    b.* = try LayerTensor.alloc(context, &.{output}, .{
                        .is_complex = extra_params.use_complex_numbers,
                    });
                    bias_created += 1;
                }
            }

            return Layer{
                .vtable = .{
                    .deinit = &deinit,
                    .getCachedOutput = &getCachedOutput,
                    .getWeights = &getWeights,
                    .getBias = &getBias,
                    .prepareCache = &prepareCache,
                    .releaseCache = &releaseCache,
                    .forward = &forward,
                    .getSensitivity = &getSensitivity,
                    .backward = &backward,
                    .getGradients = &getGradients,
                    .getBiasGradients = &getBiasGradients,
                },
                .ptr = layer,
            };
        }

        pub fn deinit(ptr: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            for (self.weights) |w| {
                w.release();
            }
            allocator.free(self.weights);

            if (self.bias_enabled) {
                for (self.bias) |b| {
                    b.?.release();
                }
                allocator.free(self.bias);
            }

            allocator.destroy(self);
        }

        fn getCachedOutput(_: *const anyopaque, cache: *const anyopaque) *LayerTensor {
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));
            const outputs = cache_data.outputs;
            return outputs[outputs.len - 1];
        }

        fn getWeights(ptr: *const anyopaque) []const *LayerTensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            return self.weights;
        }

        fn getBias(ptr: *const anyopaque) ?[]const ?*LayerTensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (!self.bias_enabled) return null;

            return self.bias;
        }

        fn prepareCache(
            ptr: *const anyopaque,
            number_of_elements: u64,
        ) !*anyopaque {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const outputs = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(outputs);

            const sensitivities = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(sensitivities);

            const acti_derivatives = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(acti_derivatives);

            const gradients = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(gradients);

            const bias_gradients = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(bias_gradients);

            const bias_gradients_lis = try self.allocator.alloc(u64, self.weights.len);
            errdefer self.allocator.free(bias_gradients_lis);

            var slots_created: usize = 0;
            errdefer {
                for (
                    outputs[0..slots_created],
                    sensitivities[0..slots_created],
                    acti_derivatives[0..slots_created],
                    gradients[0..slots_created],
                    bias_gradients[0..slots_created],
                ) |o, s, ad, g, bg| {
                    o.release();
                    s.release();
                    ad.release();
                    g.release();
                    bg.release();
                }
            }

            const context = self.context;
            const default_command_queue = &context.command_queues[0];

            for (
                self.weights,
                outputs,
                sensitivities,
                acti_derivatives,
                gradients,
                bias_gradients,
                bias_gradients_lis,
            ) |w, *o, *s, *ad, *g, *gb, *gb_li| {
                const output = w.dimensions.shape[0];
                const use_complex_numbers = w.flags.is_complex;

                o.* = try LayerTensor.alloc(context, &.{ number_of_elements, output }, .{
                    .is_complex = use_complex_numbers,
                });
                errdefer o.*.release();

                s.* = try LayerTensor.alloc(context, &.{ number_of_elements, output }, .{
                    .is_complex = use_complex_numbers,
                });
                errdefer s.*.release();

                ad.* = try LayerTensor.alloc(context, &.{ number_of_elements, output }, .{
                    .is_complex = use_complex_numbers,
                });
                errdefer ad.*.release();

                try wekua.tensor.fill.constant(T, default_command_queue, s.*, 1, null);

                g.* = try LayerTensor.alloc(context, w.dimensions.shape, .{
                    .is_complex = use_complex_numbers,
                });
                errdefer g.*.release();

                gb.* = try LayerTensor.alloc(context, &.{output}, .{
                    .is_complex = use_complex_numbers,
                });
                errdefer gb.*.release();

                wekua.utils.calculateWorkItems(
                    &.{gb.*.memory_layout.row_pitch_for_vectors},
                    @as([*]u64, @ptrCast(gb_li))[0..1],
                    default_command_queue.max_work_group_size,
                );

                slots_created += 1;
            }

            const cache = try self.allocator.create(LinearCache);
            cache.* = .{
                .outputs = outputs,
                .sensitivities = sensitivities,
                .acti_derivatives = acti_derivatives,
                .gradients = gradients,
                .bias_gradients = bias_gradients,
                .bias_gradients_lis = bias_gradients_lis,
            };

            return cache;
        }

        fn releaseCache(ptr: *const anyopaque, cache: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));

            for (cache_data.outputs) |o| {
                o.release();
            }

            for (cache_data.sensitivities) |s| {
                s.release();
            }

            for (cache_data.acti_derivatives) |ad| {
                ad.release();
            }

            for (cache_data.gradients) |g| {
                g.release();
            }

            for (cache_data.bias_gradients) |bg| {
                bg.release();
            }

            allocator.free(cache_data.outputs);
            allocator.free(cache_data.sensitivities);
            allocator.free(cache_data.acti_derivatives);
            allocator.free(cache_data.gradients);
            allocator.free(cache_data.bias_gradients);
            allocator.free(cache_data.bias_gradients_lis);

            allocator.destroy(cache_data);
        }

        fn addBias(
            command_queue: *const CommandQueue,
            output: *LayerTensor,
            bias: *LayerTensor,
        ) !void {
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                output,
                .LinearBias,
                "bias",
                bias_cl_kernel,
                null,
            );

            const prev_events = output.events_manager.getPrevEvents(.write);

            const set_arg = cl.kernel.set_arg;
            const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(&output.buffer));
            try set_arg(kernel, 1, cl_mem_size, @ptrCast(&bias.buffer));

            try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&output.memory_layout.row_pitch));
            try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&output.memory_layout.slice_pitch));

            var new_event: cl.event.cl_event = undefined;
            try cl.kernel.enqueue_nd_range(
                command_queue.cmd,
                kernel,
                null,
                &output.work_configuration.global_work_items,
                &output.work_configuration.local_work_items[command_queue.wekua_id],
                prev_events,
                &new_event,
            );
            errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

            _ = try output.events_manager.appendNewEvent(.write, prev_events, new_event, null);
        }

        fn forward(
            ptr: *const anyopaque,
            command_queue: *const CommandQueue,
            input: *LayerTensor,
            cache: *anyopaque,
        ) !*LayerTensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const linear_cache: *LinearCache = @ptrCast(@alignCast(cache));
            const outputs_cached = linear_cache.outputs;

            var in = input;

            const bias = self.bias;
            const bias_enabled = self.bias_enabled;

            const activation_layer = self.activation;

            for (self.weights, outputs_cached, 0..) |w, oc, index| {
                try wekua.blas.gemm.perform(
                    T,
                    command_queue,
                    null,
                    null,
                    in,
                    .no_transpose,
                    w,
                    .transpose,
                    null,
                    null,
                    oc,
                );

                if (bias_enabled) {
                    try addBias(command_queue, oc, bias[index].?);
                }

                if (activation_layer) |*acti| {
                    try acti.run(command_queue, oc);
                }

                in = oc;
            }

            return in;
        }

        fn getSensitivity(_: *const anyopaque, cache: *const anyopaque) *LayerTensor {
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));

            const gradients = cache_data.sensitivities;
            return gradients[gradients.len - 1];
        }

        fn getBiasSensitivity(
            command_queue: *const core.CommandQueue,
            sensitivity: *LayerTensor,
            bias_gradient: *LayerTensor,
            lwi: []const u64,
        ) !void {
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                sensitivity,
                .LinearBiasStep,
                "bias_step",
                bias_step_cl_kernel,
                null,
            );

            const sensitivity_prev_events = sensitivity.events_manager.getPrevEvents(.read);
            const bias_gradient_prev_events = bias_gradient.events_manager.getPrevEvents(.write);

            const events_set = try wekua.tensor.EventManager.EventsSet.init(
                command_queue.allocator,
                &.{ sensitivity_prev_events, bias_gradient_prev_events },
                null,
            );
            errdefer events_set.release();

            const prev_events = events_set.getPrevEvents();

            const set_arg = cl.kernel.set_arg;
            const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(&sensitivity.buffer));
            try set_arg(kernel, 1, cl_mem_size, @ptrCast(&bias_gradient.buffer));

            try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&sensitivity.memory_layout.row_pitch_for_vectors));
            try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&sensitivity.dimensions.shape[0]));

            var new_event: cl.event.cl_event = undefined;
            try cl.kernel.enqueue_nd_range(
                command_queue.cmd,
                kernel,
                null,
                &.{bias_gradient.memory_layout.row_pitch_for_vectors},
                lwi,
                prev_events,
                &new_event,
            );
            errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

            try events_set.appendNewEvent(
                T,
                true,
                &.{ .read, .write },
                &.{ sensitivity, bias_gradient },
                prev_events,
                new_event,
            );
        }

        fn backward(
            ptr: *const anyopaque,
            command_queue: *const CommandQueue,
            cache: *anyopaque,
            input: *LayerTensor,
            input_sensitivity: ?*LayerTensor,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            const cache_data: *LinearCache = @ptrCast(@alignCast(cache));

            const weights = self.weights;
            const activation_layer = self.activation;
            const bias_enabled = self.bias_enabled;

            const acti_derivatives = cache_data.acti_derivatives;
            const sensitivities = cache_data.sensitivities;
            const bias_gradients = cache_data.bias_gradients;
            const gradients = cache_data.gradients;
            const outputs = cache_data.outputs;

            const bias_gradients_lis = cache_data.bias_gradients_lis;

            var sensitivity = sensitivities[sensitivities.len - 1];

            var index: usize = weights.len - 1;
            var output = outputs[index];
            while (true) {
                const acti_derivative = acti_derivatives[index];
                if (activation_layer) |*acti| {
                    try acti.getDerivative(command_queue, output, acti_derivative);
                }

                try wekua.math.basic.dot(T, command_queue, sensitivity, acti_derivative);

                const prev_output: *LayerTensor = blk: {
                    if (index >= 1) {
                        break :blk outputs[index - 1];
                    }
                    break :blk input;
                };
                output = prev_output;

                try wekua.blas.gemm.perform(
                    T,
                    command_queue,
                    null,
                    null,
                    sensitivity,
                    .transpose,
                    prev_output,
                    .no_transpose,
                    null,
                    null,
                    gradients[index],
                );

                if (bias_enabled) {
                    try getBiasSensitivity(
                        command_queue,
                        sensitivity,
                        bias_gradients[index],
                        bias_gradients_lis[index..(index + 1)],
                    );
                }

                const next_sensitivity: *LayerTensor = blk: {
                    if (index >= 1) {
                        break :blk sensitivities[index - 1];
                    }

                    break :blk input_sensitivity orelse return;
                };

                try wekua.blas.gemm.perform(
                    T,
                    command_queue,
                    null,
                    null,
                    sensitivity,
                    .no_transpose,
                    weights[index],
                    .no_transpose,
                    null,
                    null,
                    next_sensitivity,
                );

                if (index == 0) break;

                index -= 1;
                sensitivity = next_sensitivity;
            }
        }

        fn getGradients(_: *const anyopaque, cache: *const anyopaque) []const *LayerTensor {
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));
            return cache_data.gradients;
        }

        fn getBiasGradients(ptr: *const anyopaque, cache: *const anyopaque) ?[]const *LayerTensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            if (!self.bias_enabled) return null;

            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));
            return cache_data.bias_gradients;
        }
    };
}
