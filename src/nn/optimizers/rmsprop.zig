const std = @import("std");
const cl = @import("opencl");

const wekua = @import("../../wekua.zig");

const KernelsSet = wekua.core.KernelsSet;

const optimizer = @import("main.zig");
const w_layer = @import("../layer/main.zig");

const rmsprop_cl_kernel = @embedFile("kernels/rmsprop.cl");

// Root Mean Square Propagation
pub fn RMSProp(comptime T: type) type {
    switch (T) {
        f32, f64 => {},
        else => @compileError("RMSProp optimizer only supports f32 and f64 types"),
    }

    const Cache = w_layer.Cache(T);
    const Optimizer = optimizer.Optimizer(T);
    const Tensor = wekua.Tensor(T);

    return struct {
        pub const Config = struct {
            lr: T = 0.001,
            lri: T = 0,
            gamma: T = 0.9,
            gammai: T = 0,
        };

        allocator: std.mem.Allocator,
        gradient_histories: []*Tensor,
        bias_gradient_histories: []*Tensor,
        config: Config,

        const Self = @This();

        pub fn init(context: *const wekua.core.Context, cache: *const Cache, config: Config) !Optimizer {
            const allocator = context.allocator;
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            var gradient_histories_array = std.ArrayList(*Tensor).init(allocator);
            errdefer gradient_histories_array.deinit();

            var bias_gradient_histories_array = std.ArrayList(*Tensor).init(allocator);
            errdefer bias_gradient_histories_array.deinit();

            var items_created: usize = 0;
            errdefer {
                for (
                    gradient_histories_array.items[0..items_created],
                    bias_gradient_histories_array.items[0..items_created],
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

            return Optimizer{
                .vtable = .{
                    .step = &step,
                    .zero = &zero,
                    .deinit = &deinit,
                },
                .ptr = self,
            };
        }

        fn executeRMSProp(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            x: *Tensor,
            gradient: *Tensor,
            gradient_history: *Tensor,
        ) !void {
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                x,
                .RMSProp,
                "rmsprop_kernel",
                rmsprop_cl_kernel,
                null,
            );

            const x_prev_events = x.events_manager.getPrevEvents(.write);
            const gradient_prev_events = gradient.events_manager.getPrevEvents(.read);
            const gradient_history_prev_events = gradient_history.events_manager.getPrevEvents(.write);

            const events_set = try wekua.tensor.EventManager.EventsSet.init(
                command_queue.allocator,
                &.{ x_prev_events, gradient_prev_events, gradient_history_prev_events },
                null,
            );
            errdefer events_set.release();

            const prev_events = events_set.getPrevEvents();

            const set_arg = cl.kernel.set_arg;
            const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
            try set_arg(kernel, 1, cl_mem_size, @ptrCast(&gradient.buffer));
            try set_arg(kernel, 2, cl_mem_size, @ptrCast(&gradient_history.buffer));

            try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&x.memory_layout.row_pitch_for_vectors));
            try set_arg(kernel, 4, @sizeOf(u64), @ptrCast(&x.memory_layout.slice_pitch_for_vectors));

            try set_arg(kernel, 5, @sizeOf(T), @ptrCast(&self.config.lr));
            try set_arg(kernel, 6, @sizeOf(T), @ptrCast(&self.config.gamma));

            if (x.flags.is_complex) {
                try set_arg(kernel, 7, @sizeOf(T), @ptrCast(&self.config.lri));
                try set_arg(kernel, 8, @sizeOf(T), @ptrCast(&self.config.gammai));
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
                &.{ x, gradient, gradient_history },
                new_event,
            );
        }

        fn step(
            ptr: *anyopaque,
            command_queue: *const wekua.core.CommandQueue,
            cache: *const Cache,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const gradient_histories = self.gradient_histories;
            const bias_gradient_histories = self.bias_gradient_histories;
            var index: usize = 0;

            for (cache.slots) |*slot| {
                const layer = slot.layer;
                const gradients = layer.getGradients(slot.cache);
                const weights = layer.getWeights();

                var _index = index;
                for (weights, gradients) |w, g| {
                    try self.executeRMSProp(command_queue, w, g, gradient_histories[index]);
                    _index += 1;
                }

                if (layer.getBiasGradients(slot.cache)) |bias_gradients| {
                    _index = index;
                    const bias = layer.getBias();
                    for (bias.?, bias_gradients) |b, maybe_bg| {
                        const bg = maybe_bg orelse continue;

                        try self.executeRMSProp(command_queue, b.?, bg, bias_gradient_histories[index]);
                    }
                }

                index += _index;
            }
        }

        fn zero(ptr: *anyopaque, command_queue: *const wekua.core.CommandQueue) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.gradient_histories, self.bias_gradient_histories) |v, bv| {
                try wekua.tensor.fill.zeroes(T, v, command_queue);
                try wekua.tensor.fill.zeroes(T, bv, command_queue);
            }
        }

        fn deinit(ptr: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            for (self.gradient_histories, self.bias_gradient_histories) |v, bv| {
                v.release();
                bv.release();
            }

            const allocator = self.allocator;
            allocator.free(self.gradient_histories);
            allocator.free(self.bias_gradient_histories);
            self.allocator.destroy(self);
        }
    };
}
