const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const utils = @import("utils");

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const blas = @import("blas");
const math = @import("math");

const activation_module = @import("../activation/main.zig");
const layer_module = @import("main.zig");

const bias_cl_kernel: []const u8 = @embedFile("kernels/bias.cl");
const bias_step_cl_kernel: []const u8 = @embedFile("kernels/bias_step.cl");

pub const ExtraParams = struct {
    deep: usize = 1,
    enable_bias: bool = true,
};

inline fn getRandomLimits(comptime T: type, input: usize, output: usize, start: *T, end: *T) void {
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
    const TensorT = Tensor(T);
    const Layer = layer_module.Layer(T);
    const Activation = activation_module.Activation(T);

    const SubType = core.types.getType(T);

    const LinearCache = struct {
        outputs: []*TensorT,
        sensitivities: []*TensorT,
        acti_derivatives: []*TensorT,

        gradients: []*TensorT,
        bias_gradients: []*TensorT,
        bias_gradients_lis: []u64,
    };

    return struct {
        allocator: std.mem.Allocator,

        weights: []*TensorT,

        bias_enabled: bool,
        bias: []?*TensorT,

        activation: ?activation_module.Activation(T),

        context: *const core.Context,

        const Self = @This();

        pub fn init(
            context: *const core.Context,
            pipeline: *Pipeline,
            input: usize,
            output: usize,
            acti: ?Activation,
            extra_params: ExtraParams,
        ) TensorErrors!Layer {
            if (input == 0 or output == 0 or extra_params.deep == 0) return TensorErrors.InvalidValue;

            const allocator = context.allocator;

            const self_layer = try allocator.create(Self);
            errdefer allocator.destroy(self_layer);

            self_layer.activation = acti;
            self_layer.allocator = allocator;
            self_layer.context = context;
            self_layer.bias_enabled = extra_params.enable_bias;

            var bottom: SubType = undefined;
            var top: SubType = undefined;
            getRandomLimits(SubType, input, output, &bottom, &top);

            const weights = try allocator.alloc(*TensorT, extra_params.deep);
            self_layer.weights = weights;

            var weights_created: usize = 0;
            errdefer {
                for (weights[0..weights_created]) |w| {
                    w.release(pipeline);
                }
                allocator.free(weights);
            }

            const first_weight_layer = try TensorT.alloc(context, pipeline, &.{ output, input }, .{});
            weights[0] = first_weight_layer;
            weights_created += 1;

            try tensor_module.random.uniform(
                T,
                pipeline,
                first_weight_layer,
                null,
                bottom,
                top,
            );

            getRandomLimits(SubType, output, output, &bottom, &top);
            for (weights[1..]) |*w| {
                const weight = try TensorT.alloc(context, pipeline, &.{ output, output }, .{});
                errdefer weight.release(pipeline);

                try tensor_module.random.uniform(T, pipeline, weight, null, bottom, top);

                w.* = weight;
                weights_created += 1;
            }

            if (extra_params.enable_bias) {
                const bias_layers = try allocator.alloc(?*TensorT, extra_params.deep);
                self_layer.bias = bias_layers;

                var bias_created: usize = 0;
                errdefer {
                    for (bias_layers[0..bias_created]) |b| {
                        b.?.release(pipeline);
                    }
                    allocator.free(bias_layers);
                }

                for (bias_layers) |*b| {
                    b.* = try TensorT.alloc(context, pipeline, &.{output}, .{});
                    bias_created += 1;
                }
            }

            return Layer{
                .vtable = .{
                    .deinit = &layerDeinit,
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
                .ptr = self_layer,
            };
        }

        pub fn layerDeinit(ptr: *const anyopaque, pipeline: *Pipeline) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            for (self.weights) |w| {
                w.release(pipeline);
            }
            allocator.free(self.weights);

            if (self.bias_enabled) {
                for (self.bias) |b| {
                    b.?.release(pipeline);
                }
                allocator.free(self.bias);
            }

            allocator.destroy(self);
        }

        fn getCachedOutput(_: *const anyopaque, cache: *const anyopaque) *TensorT {
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));
            const outputs = cache_data.outputs;
            return outputs[outputs.len - 1];
        }

        fn getWeights(ptr: *const anyopaque) []const *TensorT {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            return self.weights;
        }

        fn getBias(ptr: *const anyopaque) ?[]const ?*TensorT {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (!self.bias_enabled) return null;

            return self.bias;
        }

        fn prepareCache(
            ptr: *const anyopaque,
            pipeline: *Pipeline,
            number_of_elements: u64,
        ) TensorErrors!*anyopaque {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const outputs = try self.allocator.alloc(*TensorT, self.weights.len);
            errdefer self.allocator.free(outputs);

            const sensitivities = try self.allocator.alloc(*TensorT, self.weights.len);
            errdefer self.allocator.free(sensitivities);

            const acti_derivatives = try self.allocator.alloc(*TensorT, self.weights.len);
            errdefer self.allocator.free(acti_derivatives);

            const gradients = try self.allocator.alloc(*TensorT, self.weights.len);
            errdefer self.allocator.free(gradients);

            const bias_gradients = try self.allocator.alloc(*TensorT, self.weights.len);
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
                    o.release(pipeline);
                    s.release(pipeline);
                    ad.release(pipeline);
                    g.release(pipeline);
                    bg.release(pipeline);
                }
            }

            const context = self.context;
            const command_queue = pipeline.command_queue;

            for (
                self.weights,
                outputs,
                sensitivities,
                acti_derivatives,
                gradients,
                bias_gradients,
                bias_gradients_lis,
            ) |w, *o, *s, *ad, *g, *gb, *gb_li| {
                const weight_output = w.dimensions.shape[0];

                o.* = try TensorT.alloc(context, pipeline, &.{ number_of_elements, weight_output }, .{});
                errdefer o.*.release(pipeline);

                s.* = try TensorT.alloc(context, pipeline, &.{ number_of_elements, weight_output }, .{});
                errdefer s.*.release(pipeline);

                ad.* = try TensorT.alloc(context, pipeline, &.{ number_of_elements, weight_output }, .{});
                errdefer ad.*.release(pipeline);

                const one_val: T = if (comptime core.types.isComplex(T))
                    .{ .real = 1, .imag = 0 }
                else
                    1;

                try tensor_module.fill.constant(T, pipeline, s.*, one_val);

                g.* = try TensorT.alloc(context, pipeline, w.dimensions.shape, .{});
                errdefer g.*.release(pipeline);

                gb.* = try TensorT.alloc(context, pipeline, &.{weight_output}, .{});
                errdefer gb.*.release(pipeline);

                utils.calculateWorkItems(
                    &.{gb.*.memory_layout.row_pitch_for_vectors},
                    @as([*]u64, @ptrCast(gb_li))[0..1],
                    command_queue.max_work_group_size,
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

        fn releaseCache(ptr: *const anyopaque, pipeline: *Pipeline, cache: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));

            for (cache_data.outputs) |o| {
                o.release(pipeline);
            }

            for (cache_data.gradients) |g| {
                g.release(pipeline);
            }

            for (cache_data.bias_gradients) |bg| {
                bg.release(pipeline);
            }

            for (cache_data.acti_derivatives) |ad| {
                ad.release(pipeline);
            }

            for (cache_data.sensitivities) |s| {
                s.release(pipeline);
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
            pipeline: *Pipeline,
            output: *TensorT,
            bias_tensor: *TensorT,
        ) TensorErrors!void {
            const command_queue = pipeline.command_queue;

            const vectors_enabled = output.flags.vectors_enabled;
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                vectors_enabled,
                .LinearBias,
                "bias",
                bias_cl_kernel,
                null,
            );

            const prev_events = pipeline.prevEvents();
            var row_pitch: u64 = undefined;
            var num_elements: u64 = undefined;
            var work_items: u64 = undefined;

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);
            const wekua_id = command_queue.wekua_id;

            if (vectors_enabled) {
                row_pitch = output.memory_layout.row_pitch_for_vectors;
                num_elements = output.memory_layout.number_of_vectors;
                work_items = output.work_configuration.local_work_items_for_vectors_1d[wekua_id];
            }else{
                row_pitch = output.memory_layout.row_pitch;
                num_elements = output.dimensions.number_of_elements;
                work_items = output.work_configuration.local_work_items_1d[wekua_id];
            }

            try setArg(kernel, 0, cl_mem_size, @ptrCast(&output.buffer));
            try setArg(kernel, 1, cl_mem_size, @ptrCast(&bias_tensor.buffer));
            try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&row_pitch));

            var new_event: cl.event.Event = undefined;
            try cl.kernel.enqueueNdRange(
                command_queue.cl_command_queue,
                kernel,
                null,
                &.{num_elements},
                &.{work_items},
                prev_events,
                &new_event,
            );
            errdefer tensor_module.helpers.releaseEvent(new_event);

            try pipeline.append(&.{new_event});
        }

        fn forward(
            ptr: *const anyopaque,
            pipeline: *Pipeline,
            input_tensor: *TensorT,
            cache: *anyopaque,
        ) TensorErrors!*TensorT {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const linear_cache: *LinearCache = @ptrCast(@alignCast(cache));
            const outputs_cached = linear_cache.outputs;

            var input = input_tensor;

            const bias_slice = self.bias;
            const bias_enabled = self.bias_enabled;

            const activation_layer = self.activation;

            for (self.weights, outputs_cached, 0..) |weight, output, index| {
                try blas.gemm(
                    T,
                    pipeline,
                    null,
                    input,
                    .no_transpose,
                    weight,
                    .transpose,
                    null,
                    output,
                );

                // TODO: avoid this repeated condition
                if (bias_enabled) {
                    try addBias(pipeline, output, bias_slice[index].?);
                }

                if (activation_layer) |*act| {
                    try act.run(pipeline, output);
                }

                input = output;
            }

            return input;
        }

        fn getSensitivity(_: *const anyopaque, cache: *const anyopaque) *TensorT {
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));

            const sensitivities_slice = cache_data.sensitivities;
            return sensitivities_slice[sensitivities_slice.len - 1];
        }

        fn getBiasSensitivity(
            pipeline: *Pipeline,
            sensitivity: *TensorT,
            bias_gradient: *TensorT,
            lwi: []const u64,
        ) TensorErrors!void {
            const command_queue = pipeline.command_queue;

            const vectors_enabled = sensitivity.flags.vectors_enabled;
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                vectors_enabled,
                .LinearBiasStep,
                "bias_step",
                bias_step_cl_kernel,
                null,
            );

            const prev_events = pipeline.prevEvents();

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);

            try setArg(kernel, 0, cl_mem_size, @ptrCast(&sensitivity.buffer));
            try setArg(kernel, 1, cl_mem_size, @ptrCast(&bias_gradient.buffer));

            try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&sensitivity.memory_layout.row_pitch_for_vectors));
            try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&sensitivity.dimensions.shape[0]));

            var new_event: cl.event.Event = undefined;
            try cl.kernel.enqueueNdRange(
                command_queue.cl_command_queue,
                kernel,
                null,
                @as([*]const u64, @ptrCast(&bias_gradient.memory_layout.row_pitch_for_vectors))[0..1],
                lwi,
                prev_events,
                &new_event,
            );
            errdefer tensor_module.helpers.releaseEvent(new_event);

            try pipeline.append(&.{new_event});
        }

        fn backward(
            ptr: *const anyopaque,
            pipeline: *Pipeline,
            cache: *anyopaque,
            input_tensor: *TensorT,
            input_sensitivity: ?*TensorT,
        ) TensorErrors!void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            const cache_data: *LinearCache = @ptrCast(@alignCast(cache));

            const weights = self.weights;
            const activation_layer = self.activation;
            const bias_enabled = self.bias_enabled;

            const acti_derivatives = cache_data.acti_derivatives;
            const sensitivities = cache_data.sensitivities;
            const bias_grads = cache_data.bias_gradients;
            const grads = cache_data.gradients;
            const outs = cache_data.outputs;

            const bias_gradients_lis = cache_data.bias_gradients_lis;

            var sensitivity = sensitivities[sensitivities.len - 1];

            var index: usize = weights.len - 1;
            var output = outs[index];
            while (true) {
                const acti_derivative = acti_derivatives[index];
                if (activation_layer) |*act| {
                    try act.getDerivative(pipeline, output, acti_derivative);
                }

                try math.basic.dot(T, pipeline, sensitivity, acti_derivative);

                const prev_output: *TensorT = blk: {
                    if (index >= 1) {
                        break :blk outs[index - 1];
                    }
                    break :blk input_tensor;
                };
                output = prev_output;

                try blas.gemm(
                    T,
                    pipeline,
                    null,
                    sensitivity,
                    .transpose,
                    prev_output,
                    .no_transpose,
                    null,
                    grads[index],
                );

                if (bias_enabled) {
                    try getBiasSensitivity(
                        pipeline,
                        sensitivity,
                        bias_grads[index],
                        bias_gradients_lis[index..(index + 1)],
                    );
                }

                const next_sensitivity: *TensorT = blk: {
                    if (index >= 1) {
                        break :blk sensitivities[index - 1];
                    }

                    break :blk input_sensitivity orelse return;
                };

                try blas.gemm(
                    T,
                    pipeline,
                    null,
                    sensitivity,
                    .no_transpose,
                    weights[index],
                    .no_transpose,
                    null,
                    next_sensitivity,
                );

                if (index == 0) break;

                index -= 1;
                sensitivity = next_sensitivity;
            }
        }

        fn getGradients(_: *const anyopaque, cache: *const anyopaque) []const *TensorT {
            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));
            return cache_data.gradients;
        }

        fn getBiasGradients(ptr: *const anyopaque, cache: *const anyopaque) ?[]const ?*TensorT {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            if (!self.bias_enabled) return null;

            const cache_data: *const LinearCache = @ptrCast(@alignCast(cache));
            return cache_data.bias_gradients;
        }
    };
}
