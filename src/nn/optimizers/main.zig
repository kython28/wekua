const layer = @import("../layer/main.zig");

pub fn Optimizer(comptime T: type) type {
    const OptimizerLayer = layer.Layer(T);
    const OptimizerCache = OptimizerLayer.Cache;

    return struct {
        pub const VTable = struct {
            step: *const fn (ptr: *anyopaque, cache: *OptimizerCache) void,
            zero: *const fn (ptr: *anyopaque) void,
            deinit: *const fn (ptr: *anyopaque) void,
        };

        ptr: *anyopaque,

        const Self = @This();

        pub inline fn step(self: *Self, cache: *const OptimizerCache) void {
            self.vtable.step(@ptrCast(self.ptr), cache);
        }

        pub inline fn zero(self: *Self) void {
            self.vtable.zero(@ptrCast(self.ptr));
        }

        pub inline fn deinit(self: *Self) void {
            self.vtable.deinit(@ptrCast(self.ptr));
        }
    };
}
