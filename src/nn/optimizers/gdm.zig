const std = @import("std");
const cl = @import("opencl");

const wekua = @import("../../wekua.zig");

const KernelsSet = wekua.core.KernelsSet;

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
    const Optimizer = optimizer.Optimizer(T);
    const Tensor = wekua.Tensor(T);

    return struct {
        pub const Config = struct {
            lr: T,
            lri: T = 0,
            beta: T = 0.9,
            betai: T = 0,
        };

        allocator: std.mem.Allocator,
        velocities: []*Tensor,
        bias_velocities: []*Tensor,
        config: Config,

        const Self = @This();

        pub fn init(context: *const wekua.core.Context, cache: *const Cache, config: Config) !Optimizer {
            const allocator = context.allocator;
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            var velocities_array = std.ArrayList(*Tensor).init(allocator);
            errdefer velocities_array.deinit();

            var bias_velocities_array = std.ArrayList(*Tensor).init(allocator);
            errdefer bias_velocities_array.deinit();

            var velocities_created: usize = 0;
            errdefer {
                for (
                    velocities_array.items[0..velocities_created],
                    bias_velocities_array.items[0..velocities_created],
                ) |v1, v2| {
                    v1.release();
                    v2.release();
                }
            }

            for (cache.slots) |slot| {
                const layer = slot.layer;
                const weights = layer.getWeights();

                for (weights) |w| {
                    const v = try Tensor.alloc(context, w.dimensions.shape, .{
                        .is_complex = w.flags.is_complex,
                        .vectors_enabled = w.flags.vectors_enabled,
                    });
                    errdefer v.release();

                    const bv = try Tensor.alloc(context, w.dimensions.shape[0..1], .{
                        .is_complex = w.flags.is_complex,
                        .vectors_enabled = w.flags.vectors_enabled,
                    });
                    errdefer bv.release();

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

            return Optimizer{
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
            command_queue: *const wekua.core.CommandQueue,
            x: *Tensor,
            gradient: *Tensor,
            velocity: *Tensor,
        ) !void {
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                x,
                .GDM,
                "gdm_kernel",
                gdm_cl_kernel,
                null,
            );

            const x_prev_events = x.events_manager.getPrevEvents(.write);
            const gradient_prev_events = gradient.events_manager.getPrevEvents(.read);
            const velocity_prev_events = velocity.events_manager.getPrevEvents(.write);

            const events_set = try wekua.tensor.EventManager.EventsSet.init(
                command_queue.allocator,
                &.{ x_prev_events, gradient_prev_events, velocity_prev_events },
                null,
            );
            errdefer events_set.release();

            const prev_events = events_set.getPrevEvents();

            const set_arg = cl.kernel.set_arg;
            const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
            try set_arg(kernel, 1, cl_mem_size, @ptrCast(&gradient.buffer));
            try set_arg(kernel, 2, cl_mem_size, @ptrCast(&velocity.buffer));

            try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&x.memory_layout.row_pitch_for_vectors));
            try set_arg(kernel, 4, @sizeOf(u64), @ptrCast(&x.memory_layout.slice_pitch_for_vectors));

            try set_arg(kernel, 5, @sizeOf(T), @ptrCast(&self.config.lr));
            try set_arg(kernel, 6, @sizeOf(T), @ptrCast(&self.config.beta));

            if (x.flags.is_complex) {
                try set_arg(kernel, 7, @sizeOf(T), @ptrCast(&self.config.lri));
                try set_arg(kernel, 8, @sizeOf(T), @ptrCast(&self.config.betai));
            }

            var new_event: cl.event.cl_event = undefined;
            try cl.kernel.enqueue_nd_range(
                command_queue.cmd,
                kernel,
                null,
                &x.work_configuration.global_work_items,
                &x.work_configuration.local_work_items[command_queue.wekua_id],
                prev_events,
                &new_event,
            );

            try events_set.appendNewEvent(
                T,
                true,
                &.{ .write, .read, .write },
                &.{ x, gradient, velocity },
                new_event,
            );
        }

        fn step(
            ptr: *anyopaque,
            command_queue: *const wekua.core.CommandQueue,
            cache: *const Cache,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const velocities = self.velocities;
            const bias_velocities = self.bias_velocities;
            var index: usize = 0;

            for (cache.slots) |*slot| {
                const layer = slot.layer;
                const gradients = layer.getGradients(slot.cache);
                const weights = layer.getWeights();

                var _index = index;
                for (weights, gradients) |w, g| {
                    try self.executeGDM(command_queue, w, g, velocities[index]);
                    _index += 1;
                }

                if (layer.getBiasGradients(slot.cache)) |bias_gradients| {
                    _index = index;
                    const bias = layer.getBias();
                    for (bias.?, bias_gradients) |b, maybe_bg| {
                        const bg = maybe_bg orelse continue;

                        try self.executeGDM(command_queue, b.?, bg, bias_velocities[index]);
                    }
                }

                index += _index;
            }
        }

        fn zero(ptr: *anyopaque, command_queue: *const wekua.core.CommandQueue) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.velocities, self.bias_velocities) |v, bv| {
                try wekua.tensor.fill.zeroes(T, v, command_queue);
                try wekua.tensor.fill.zeroes(T, bv, command_queue);
            }
        }

        fn deinit(ptr: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.velocities, self.bias_velocities) |v, bv| {
                v.release();
                bv.release();
            }

            const allocator = self.allocator;
            allocator.free(self.velocities);
            allocator.free(self.bias_velocities);

            allocator.destroy(self);
        }
    };
}
