const std = @import("std");

const wekua = @import("../../wekua.zig");

const optimizer = @import("main.zig");
const w_layer = @import("../layer/main.zig");

// Gradient Descent
pub fn GD(comptime T: type) type {
    const OptimizerLayer = w_layer.Layer(T);
    const OptimizerCache = OptimizerLayer.Cache;

    const Optimizer = optimizer.Optimizer(T);

    return struct {
        allocator: std.mem.Allocator,
        lr: ?T,
        lri: ?T,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, lr: ?T, lri: ?T) !Optimizer {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .allocator = allocator,
                .lr = lr,
                .lri = lri,
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
                    for (bias.?, bias_gradients) |b, bg| {
                        try wekua.blas.axpy(
                            T,
                            command_queue,
                            bg,
                            lr,
                            lri,
                            b,
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
