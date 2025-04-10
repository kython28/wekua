const wekua = @import("../../wekua.zig");
const layer = @import("../layer/main.zig");

const w_gd = @import("gd.zig");
const w_gdm = @import("gdm.zig");
const w_adagrad = @import("adagrad.zig");
const w_rmsprop = @import("rmsprop.zig");

pub fn Optimizer(comptime T: type) type {
    const Cache = layer.Cache(T);

    return struct {
        pub const GD = w_gd.GD(T);
        pub const GDM = w_gdm.GDM(T);
        pub const Adagrad = w_adagrad.Adagrad(T);
        pub const RMSProp = w_rmsprop.RMSProp(T);

        pub const VTable = struct {
            step: *const fn (
                ptr: *anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                cache: *const Cache,
            ) anyerror!void,
            zero: *const fn (ptr: *anyopaque, command_queue: *const wekua.core.CommandQueue) anyerror!void,
            deinit: *const fn (ptr: *anyopaque) void,
        };

        vtable: VTable,
        ptr: *anyopaque,

        const Self = @This();

        pub inline fn step(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            cache: *const Cache,
        ) !void {
            try self.vtable.step(self.ptr, command_queue, cache);
        }

        pub inline fn zero(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
        ) !void {
            try self.vtable.zero(self.ptr, command_queue);
        }

        pub inline fn deinit(self: *const Self) void {
            self.vtable.deinit(self.ptr);
        }
    };
}
