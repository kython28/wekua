const wekua = @import("../../wekua.zig");
const layer = @import("../layer/main.zig");

const w_gd = @import("gd.zig");

pub fn Optimizer(comptime T: type) type {
    const OptimizerCache = layer.Cache(T);

    return struct {
        pub const GD = w_gd.GD(T);

        pub const VTable = struct {
            step: *const fn (
                ptr: *anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                cache: *const OptimizerCache,
            ) anyerror!void,
            zero: *const fn (ptr: *anyopaque) anyerror!void,
            deinit: *const fn (ptr: *anyopaque) void,
        };

        vtable: VTable,
        ptr: *anyopaque,

        const Self = @This();

        pub inline fn step(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            cache: *const OptimizerCache,
        ) !void {
            try self.vtable.step(@ptrCast(self.ptr), command_queue, cache);
        }

        pub inline fn zero(self: *const Self) !void {
            try self.vtable.zero(@ptrCast(self.ptr));
        }

        pub inline fn deinit(self: *const Self) void {
            self.vtable.deinit(@ptrCast(self.ptr));
        }
    };
}
