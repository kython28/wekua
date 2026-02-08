const core = @import("core");
const Pipeline = core.Pipeline;

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
                pipeline: *Pipeline,
                cache: *const Cache,
            ) anyerror!void,
            zero: *const fn (ptr: *anyopaque, pipeline: *Pipeline) anyerror!void,
            deinit: *const fn (ptr: *anyopaque, pipeline: *Pipeline) void,
        };

        vtable: VTable,
        ptr: *anyopaque,

        const Self = @This();

        pub inline fn step(
            self: *const Self,
            pipeline: *Pipeline,
            cache: *const Cache,
        ) !void {
            try self.vtable.step(self.ptr, pipeline, cache);
        }

        pub inline fn zero(
            self: *const Self,
            pipeline: *Pipeline,
        ) !void {
            try self.vtable.zero(self.ptr, pipeline);
        }

        pub inline fn deinit(self: *const Self, pipeline: *Pipeline) void {
            self.vtable.deinit(self.ptr, pipeline);
        }
    };
}

test {
    _ = w_gd;
    _ = w_gdm;
    _ = w_adagrad;
    _ = w_rmsprop;
}
