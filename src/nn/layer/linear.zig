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
        gradients: []*LayerTensor,
        acti_gradients: []*LayerTensor,
    };

    return struct {
        allocator: std.mem.Allocator,

        weights: []*LayerTensor,

        bias_enabled: bool,
        bias: []*LayerTensor,

        activation: ?w_activation.Activation(T),

        context: *const wekua.core.Context,

        const Self = @This();

        pub fn init(
            context: *const wekua.core.Context,
            input: usize,
            output: usize,
            deep: usize,
            enable_bias: bool,
            activation: ?w_activation.Activation(T),
        ) !Layer {
            const allocator = context.allocator;

            const layer = try allocator.create(Self);
            errdefer allocator.destroy(layer);

            layer.activation = activation;

            var bottom: T = undefined;
            var top: T = undefined;
            get_random_limits(T, input, output, &bottom, &top);

            const weights = try allocator.alloc(*LayerTensor, deep);
            layer.weights = weights;

            var weights_created: usize = 0;
            errdefer {
                for (weights[0..weights_created]) |w| {
                    w.release();
                }
                allocator.free(weights);
            }

            const first_weight_layer = try LayerTensor.alloc(context, &.{ output, input }, .{});
            weights[0] = first_weight_layer;
            weights_created += 1;

            const default_command_queue = context.command_queues[0];
            try wekua.tensor.random.uniform(
                T,
                default_command_queue.command_queue,
                first_weight_layer,
                null,
                bottom,
                top,
            );

            get_random_limits(T, output, output, &bottom, &top);
            for (weights[1..]) |*w| {
                const weight = try LayerTensor.alloc(context, &.{ output, output }, .{});
                errdefer weight.release();

                try wekua.tensor.random.uniform(T, default_command_queue, weight, null, bottom, top);

                w.* = weight;
                weights_created += 1;
            }

            if (enable_bias) {
                const bias_layers = try allocator.alloc(*LayerTensor, deep);
                layer.bias = bias_layers;

                var bias_created: usize = 0;
                errdefer {
                    if (bias_layers) |bl| {
                        for (bl[0..bias_created]) |b| {
                            b.release();
                        }
                        allocator.free(bl);
                    }
                }

                for (bias_layers) |*b| {
                    b.* = try LayerTensor.alloc(context, &.{output}, .{});
                    bias_created += 1;
                }
            }

            return Layer{
                .vtable = .{
                    .deinit = &deinit,
                    .prepareCache = &prepareCache,
                    .releaseCache = &releaseCache,
                    .forward = &forward,
                    .backward = &backward,
                },
                .ptr = layer,
                .context = context,
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
                    b.release();
                }
                allocator.free(self.bias);
            }

            allocator.destroy(self);
        }

        fn prepareCache(
            ptr: *const anyopaque,
            number_of_elements: u64,
        ) !*anyopaque {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const outputs = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(outputs);

            const gradients = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(gradients);

            const acti_gradients = try self.allocator.alloc(*LayerTensor, self.weights.len);
            errdefer self.allocator.free(acti_gradients);

            var slots_created: usize = 0;
            errdefer {
                for (
                    outputs[0..slots_created],
                    gradients[0..slots_created],
                    acti_gradients[0..slots_created],
                ) |o, g, ag| {
                    o.release();
                    g.release();
                    ag.release();
                }
            }

            const context = self.context;
            const default_command_queue = context.command_queues[0];

            for (self.weights, outputs, gradients, acti_gradients) |w, *o, *g, *ag| {
                const output = w.shape[0];

                o.* = try LayerTensor.alloc(context, &.{ number_of_elements, output }, .{});
                errdefer o.*.release();

                g.* = try LayerTensor.alloc(context, &.{ number_of_elements, output }, .{});
                errdefer g.*.release();

                ag.* = try LayerTensor.alloc(context, &.{ number_of_elements, output }, .{});
                errdefer ag.*.release();

                try wekua.tensor.fill(T, default_command_queue, g.*, 1, null);

                slots_created += 1;
            }

            const cache = try self.allocator.create(LinearCache);
            cache.* = .{
                .outputs = outputs,
                .gradients = gradients,
                .acti_gradients = acti_gradients,
            };

            return cache;
        }

        fn releaseCache(ptr: *const anyopaque, cache: *anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            const cache_data: *LinearCache = @ptrCast(cache);

            for (cache_data.outputs) |o| {
                o.release();
            }

            for (cache_data.gradients) |g| {
                g.release();
            }

            for (cache_data.acti_gradients) |ag| {
                ag.release();
            }

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
            cache: *LinearCache,
        ) !*LayerTensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const outputs_cached = cache.outputs;

            var in = input;

            const bias_enabled = self.bias_enabled;
            const activation_layer = self.activation;

            for (self.weights, self.bias, outputs_cached) |w, b, oc| {
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
                    try self.addBias(command_queue, oc, b);
                }

                if (activation_layer) |*acti| {
                    acti.run(command_queue, oc);
                }

                in = oc;
            }

            return in;
        }

        fn getGradient(_: *const anyopaque, cache: *anyopaque) *LayerTensor {
            const cache_data: *LinearCache = @ptrCast(@alignCast(cache));

            const gradients = cache_data.gradients;
            return gradients[gradients.len - 1];
        }

        fn backward(
            ptr: *const anyopaque,
            command_queue: *const CommandQueue,
            cache: *LinearCache,
            input_gradient: ?*LayerTensor,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            const cache_data: *LinearCache = @ptrCast(@alignCast(cache));

            const weights = self.weights;
            const activation_layer = self.activation;

            const acti_gradients = cache_data.acti_gradients;
            const gradients = cache_data.gradients;

            var gradient = gradients[gradients.len - 1];

            var layer_num: usize = weights.len;
            while (layer_num > 0) : (layer_num -= 1) {
                const acti_gradient = acti_gradients[layer_num - 1];
                const output = weights[layer_num - 1];
                if (activation_layer) |*acti| {
                    acti.getDerivative(command_queue, output, acti_gradient);
                }

                try wekua.math.basic.dot(T, command_queue, gradient, acti_gradient);

                const next_gradient: *LayerTensor = blk: {
                    if (layer_num > 1) {
                        break :blk gradients[layer_num - 2];
                    }

                    break :blk input_gradient orelse return;
                };

                try wekua.blas.gemm.perform(
                    T,
                    command_queue,
                    null,
                    null,
                    gradient,
                    .no_transpose,
                    weights[layer_num - 1],
                    .no_transpose,
                    null,
                    null,
                    next_gradient,
                );

                layer_num -= 1;
                gradient = next_gradient;
            }
        }
    };
}
