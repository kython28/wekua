const std = @import("std");

const wekua = @import("../../wekua.zig");

const optimizer = @import("main.zig");
const w_layer = @import("../layer/main.zig");

// Gradient Descent
pub fn GD(comptime T: type) type {
    switch (T) {
        f32, f64 => {},
        else => @compileError("Gradient Descent optimizer only supports f32 and f64 types"),
    }

    const Cache = w_layer.Cache(T);
    const Optimizer = optimizer.Optimizer(T);

    return struct {
        pub const Config = struct {
            lr: T,
            lri: T = 0,
        };

        allocator: std.mem.Allocator,
        config: Config,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, config: Config) !Optimizer {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .allocator = allocator,
                .config = .{
                    .lr = -config.lr,
                    .lri = -config.lri,
                },
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
            cache: *const Cache,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const lr = self.config.lr;
            const lri = self.config.lri;

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

        fn zero(_: *anyopaque, _: *const wekua.core.CommandQueue) !void {}

        fn deinit(ptr: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            self.allocator.destroy(self);
        }
    };
}
