const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const optimizer_module = @import("main.zig");
const layer_module = @import("../layer/main.zig");

const adagrad_cl_kernel = @embedFile("kernels/adagrad.cl");

// Adaptive gradient
pub fn Adagrad(comptime T: type) type {
    switch (T) {
        f32, f64 => {},
        else => @compileError("Adagrad optimizer only supports f32 and f64 types"),
    }

    const Cache = layer_module.Cache(T);
    const OptimizerT = optimizer_module.Optimizer(T);
    const TensorT = Tensor(T);

    return struct {
        pub const Config = struct {
            lr: T,
        };

        allocator: std.mem.Allocator,
        gradient_histories: []*TensorT,
        bias_gradient_histories: []*TensorT,
        config: Config,

        const Self = @This();

        pub fn init(
            context: *const core.Context,
            pipeline: *Pipeline,
            cache: *const Cache,
            config: Config,
        ) TensorErrors!OptimizerT {
            const allocator = context.allocator;
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            var gradient_histories_array = std.ArrayList(*TensorT).init(allocator);
            errdefer gradient_histories_array.deinit();

            var bias_gradient_histories_array = std.ArrayList(*TensorT).init(allocator);
            errdefer bias_gradient_histories_array.deinit();

            var items_created: usize = 0;
            errdefer {
                for (
                    gradient_histories_array.items[0..items_created],
                    bias_gradient_histories_array.items[0..items_created],
                ) |v1, v2| {
                    v1.release(pipeline);
                    v2.release(pipeline);
                }
            }

            for (cache.slots) |slot| {
                const layer_ref = slot.layer;
                const weights = layer_ref.getWeights();

                for (weights) |w| {
                    const v = try TensorT.alloc(context, pipeline, w.dimensions.shape, .{});
                    errdefer v.release(pipeline);

                    const bv = try TensorT.alloc(context, pipeline, w.dimensions.shape[0..1], .{});
                    errdefer bv.release(pipeline);

                    try gradient_histories_array.append(v);
                    errdefer _ = gradient_histories_array.pop();

                    try bias_gradient_histories_array.append(bv);

                    items_created += 1;
                }
            }

            const gradient_histories = try gradient_histories_array.toOwnedSlice();
            errdefer allocator.free(gradient_histories);

            const bias_gradient_histories = try bias_gradient_histories_array.toOwnedSlice();
            errdefer allocator.free(bias_gradient_histories);

            self.* = .{
                .allocator = allocator,
                .config = config,
                .gradient_histories = gradient_histories,
                .bias_gradient_histories = bias_gradient_histories,
            };

            return OptimizerT{
                .vtable = .{
                    .step = &step,
                    .zero = &zero,
                    .deinit = &deinit,
                },
                .ptr = self,
            };
        }

        fn executeAdagrad(
            self: *const Self,
            pipeline: *Pipeline,
            x: *TensorT,
            gradient: *TensorT,
            gradient_history: *TensorT,
        ) TensorErrors!void {
            const command_queue = pipeline.command_queue;

            const vectors_enabled = x.flags.vectors_enabled and gradient.flags.vectors_enabled and gradient_history.flags.vectors_enabled;
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                vectors_enabled,
                .Adagrad,
                "adagrad_kernel",
                adagrad_cl_kernel,
                null,
            );

            const prev_events = pipeline.prevEvents();

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);

            try setArg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
            try setArg(kernel, 1, cl_mem_size, @ptrCast(&gradient.buffer));
            try setArg(kernel, 2, cl_mem_size, @ptrCast(&gradient_history.buffer));
            try setArg(kernel, 3, @sizeOf(T), @ptrCast(&self.config.lr));

            var num_elements: u64 = undefined;
            var work_items: u64 = undefined;

            if (vectors_enabled) {
                num_elements = x.memory_layout.number_of_vectors;
                work_items = x.work_configuration.local_work_items_for_vectors_1d[command_queue.wekua_id];
            } else {
                num_elements = x.dimensions.number_of_elements;
                work_items = x.work_configuration.local_work_items_1d[command_queue.wekua_id];
            }

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

        fn step(
            ptr: *anyopaque,
            pipeline: *Pipeline,
            cache: *const Cache,
        ) TensorErrors!void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const gradient_histories = self.gradient_histories;
            const bias_gradient_histories = self.bias_gradient_histories;
            var index: usize = 0;

            for (cache.slots) |*slot| {
                const layer_ref = slot.layer;
                const gradients = layer_ref.getGradients(slot.cache);
                const weights = layer_ref.getWeights();

                var _index = index;
                for (weights, gradients) |w, g| {
                    try self.executeAdagrad(pipeline, w, g, gradient_histories[index]);
                    _index += 1;
                }

                if (layer_ref.getBiasGradients(slot.cache)) |bias_gradients| {
                    _index = index;
                    const bias_slice = layer_ref.getBias();
                    for (bias_slice.?, bias_gradients) |b, maybe_bg| {
                        const bg = maybe_bg orelse continue;

                        try self.executeAdagrad(pipeline, b.?, bg, bias_gradient_histories[index]);
                    }
                }

                index += _index;
            }
        }

        fn zero(ptr: *anyopaque, pipeline: *Pipeline) TensorErrors!void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.gradient_histories, self.bias_gradient_histories) |v, bv| {
                try tensor_module.fill.zeroes(T, pipeline, v);
                try tensor_module.fill.zeroes(T, pipeline, bv);
            }
        }

        fn deinit(ptr: *anyopaque, pipeline: *Pipeline) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.gradient_histories, self.bias_gradient_histories) |v, bv| {
                v.release(pipeline);
                bv.release(pipeline);
            }

            const allocator = self.allocator;
            allocator.free(self.gradient_histories);
            allocator.free(self.bias_gradient_histories);
            self.allocator.destroy(self);
        }
    };
}
