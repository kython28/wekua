const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;

const optimizer = @import("main.zig");
const w_layer = @import("../layer/main.zig");

const gdm_cl_kernel = @embedFile("kernels/gdm.cl");

// Gradient Descent Momentum
pub fn GDM(comptime T: type) type {
    switch (T) {
        f32, f64 => {},
        else => @compileError("Gradient Descent Momentum optimizer only supports f32 and f64 types"),
    }

    const Cache = w_layer.Cache(T);
    const OptimizerT = optimizer.Optimizer(T);
    const TensorT = Tensor(T);

    return struct {
        pub const Config = struct {
            lr: T,
            beta: T = 0.9,
        };

        allocator: std.mem.Allocator,
        velocities: []*TensorT,
        bias_velocities: []*TensorT,
        config: Config,

        const Self = @This();

        pub fn init(context: *const core.Context, pipeline: *Pipeline, cache: *const Cache, config: Config) !OptimizerT {
            const allocator = context.allocator;
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            var velocities_array = std.ArrayList(*TensorT).init(allocator);
            errdefer velocities_array.deinit();

            var bias_velocities_array = std.ArrayList(*TensorT).init(allocator);
            errdefer bias_velocities_array.deinit();

            var velocities_created: usize = 0;
            errdefer {
                for (
                    velocities_array.items[0..velocities_created],
                    bias_velocities_array.items[0..velocities_created],
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

                    try velocities_array.append(v);
                    errdefer _ = velocities_array.pop();

                    try bias_velocities_array.append(bv);

                    velocities_created += 1;
                }
            }

            const velocities = try velocities_array.toOwnedSlice();
            errdefer allocator.free(velocities);

            const bias_velocities = try bias_velocities_array.toOwnedSlice();
            errdefer allocator.free(bias_velocities);

            self.* = .{
                .allocator = allocator,
                .config = config,
                .velocities = velocities,
                .bias_velocities = bias_velocities,
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

        fn executeGDM(
            self: *const Self,
            pipeline: *Pipeline,
            x: *TensorT,
            gradient: *TensorT,
            velocity: *TensorT,
        ) !void {
            const command_queue = pipeline.command_queue;

            const vectors_enabled = x.flags.vectors_enabled and gradient.flags.vectors_enabled and velocity.flags.vectors_enabled;
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                vectors_enabled,
                .GDM,
                "gdm_kernel",
                gdm_cl_kernel,
                null,
            );

            const prev_events = pipeline.prevEvents();

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);

            try setArg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
            try setArg(kernel, 1, cl_mem_size, @ptrCast(&gradient.buffer));
            try setArg(kernel, 2, cl_mem_size, @ptrCast(&velocity.buffer));

            const num_elements = if (vectors_enabled)
                x.dimensions.number_of_elements
            else
                x.dimensions.number_of_elements_without_padding;

            try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&num_elements));
            try setArg(kernel, 4, @sizeOf(T), @ptrCast(&self.config.lr));
            try setArg(kernel, 5, @sizeOf(T), @ptrCast(&self.config.beta));

            var new_event: cl.event.Event = undefined;
            try cl.kernel.enqueueNdRange(
                command_queue.cl_command_queue,
                kernel,
                null,
                @ptrCast(&num_elements),
                if (vectors_enabled)
                    x.work_configuration.local_work_items_for_vectors_1d
                else
                    x.work_configuration.local_work_items_1d,
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
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const velocities = self.velocities;
            const bias_velocities = self.bias_velocities;
            var index: usize = 0;

            for (cache.slots) |*slot| {
                const layer_ref = slot.layer;
                const gradients = layer_ref.getGradients(slot.cache);
                const weights = layer_ref.getWeights();

                var _index = index;
                for (weights, gradients) |w, g| {
                    try self.executeGDM(pipeline, w, g, velocities[index]);
                    _index += 1;
                }

                if (layer_ref.getBiasGradients(slot.cache)) |bias_gradients| {
                    _index = index;
                    const bias_slice = layer_ref.getBias();
                    for (bias_slice.?, bias_gradients) |b, maybe_bg| {
                        const bg = maybe_bg orelse continue;

                        try self.executeGDM(pipeline, b.?, bg, bias_velocities[index]);
                    }
                }

                index += _index;
            }
        }

        fn zero(ptr: *anyopaque, pipeline: *Pipeline) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.velocities, self.bias_velocities) |v, bv| {
                try tensor_module.fill.zeroes(T, pipeline, v);
                try tensor_module.fill.zeroes(T, pipeline, bv);
            }
        }

        fn deinit(ptr: *anyopaque, pipeline: *Pipeline) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.velocities, self.bias_velocities) |v, bv| {
                v.release(pipeline);
                bv.release(pipeline);
            }

            const allocator = self.allocator;
            allocator.free(self.velocities);
            allocator.free(self.bias_velocities);

            allocator.destroy(self);
        }
    };
}
