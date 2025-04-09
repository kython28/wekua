const std = @import("std");

const wekua = @import("../../wekua.zig");

const optimizer = @import("main.zig");
const w_layer = @import("../layer/main.zig");

// Gradient Descent
pub fn GD(comptime T: type) type {
    const OptimizerCache = w_layer.Cache(T);

    const Optimizer = optimizer.Optimizer(T);

    return struct {
        allocator: std.mem.Allocator,
        lr: ?T,
        lri: ?T,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, lr: ?T, lri: ?T) !Optimizer {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            var lr_value: ?T = null;
            var lri_value: ?T = null;

            if (lr) |lr_value_| {
                lr_value = -lr_value_;
            }

            if (lri) |lri_value_| {
                lri_value = -lri_value_;
            }

            self.* = .{
                .allocator = allocator,
                .lr = lr_value,
                .lri = lri_value,
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

        fn step(
            ptr: *anyopaque,
            command_queue: *const wekua.core.CommandQueue,
            cache: *const OptimizerCache,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const lr = self.lr;
            const lri = self.lri;

            for (cache.slots) |*slot| {
                const layer = slot.layer;
                const gradients = layer.getGradients(slot.cache);
                const weights = layer.getWeights();

                for (weights, gradients) |w, g| {
                    try wekua.blas.axpy(
                        T,
                        command_queue,
                        g,
                        lr,
                        lri,
                        w,
                    );
                }

                if (layer.getBiasGradients(slot.cache)) |bias_gradients| {
                    const bias = layer.getBias();
                    for (bias.?, bias_gradients) |b, maybe_bg| {
                        const bg = maybe_bg orelse continue;

                        try wekua.blas.axpy(
                            T,
                            command_queue,
                            bg,
                            lr,
                            lri,
                            b.?,
                        );
                    }
                }
            }
        }

        fn zero(_: *anyopaque) !void {}

        fn deinit(ptr: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            self.allocator.destroy(self);
        }
    };
}
