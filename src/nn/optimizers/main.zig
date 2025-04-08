const wekua = @import("../../wekua.zig");
const layer = @import("../layer/main.zig");

const w_gd = @import("gd.zig");

pub fn Optimizer(comptime T: type) type {
    const OptimizerLayer = layer.Layer(T);
    const OptimizerCache = OptimizerLayer.Cache;

    return struct {
        pub const GD = w_gd.GD(T);

        pub const VTable = struct {
            step: *const fn (
                ptr: *anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                cache: *OptimizerCache,
            ) anyerror!void,
            zero: *const fn (ptr: *anyopaque) anyerror!void,
            deinit: *const fn (ptr: *anyopaque) void,
        };

        ptr: *anyopaque,

        const Self = @This();

        pub inline fn step(self: *Self, cache: *const OptimizerCache) void {
            self.vtable.step(@ptrCast(self.ptr), cache);
        }

        pub inline fn zero(self: *Self) !void {
            try self.vtable.zero(@ptrCast(self.ptr));
        }

        pub inline fn deinit(self: *Self) !void {
            try self.vtable.deinit(@ptrCast(self.ptr));
        }
    };
}
