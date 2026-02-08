const std = @import("std");

const core = @import("core");
const Pipeline = core.Pipeline;

const blas = @import("blas");

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
            pipeline: *Pipeline,
            cache: *const Cache,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const lr = self.config.lr;

            for (cache.slots) |*slot| {
                const layer_ref = slot.layer;
                const gradients = layer_ref.getGradients(slot.cache);
                const weights = layer_ref.getWeights();

                for (weights, gradients) |w, g| {
                    try blas.axpy(
                        T,
                        pipeline,
                        g,
                        lr,
                        w,
                    );
                }

                if (layer_ref.getBiasGradients(slot.cache)) |bias_gradients| {
                    const bias_slice = layer_ref.getBias();
                    for (bias_slice.?, bias_gradients) |b, maybe_bg| {
                        const bg = maybe_bg orelse continue;

                        try blas.axpy(
                            T,
                            pipeline,
                            bg,
                            lr,
                            b.?,
                        );
                    }
                }
            }
        }

        fn zero(_: *anyopaque, _: *Pipeline) !void {}

        fn deinit(ptr: *anyopaque, _: *Pipeline) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            self.allocator.destroy(self);
        }
    };
}
