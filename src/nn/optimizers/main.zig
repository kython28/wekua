const core = @import("core");
const Pipeline = core.Pipeline;

const layer = @import("../layer/main.zig");

const gd_module = @import("gd.zig");
const gdm_module = @import("gdm.zig");
const adagrad_module = @import("adagrad.zig");
const rmsprop_module = @import("rmsprop.zig");

pub fn Optimizer(comptime T: type) type {
    const Cache = layer.Cache(T);

    return struct {
        pub const GD = gd_module.GD(T);
        pub const GDM = gdm_module.GDM(T);
        pub const Adagrad = adagrad_module.Adagrad(T);
        pub const RMSProp = rmsprop_module.RMSProp(T);

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
        ) anyerror!void {
            try self.vtable.step(self.ptr, pipeline, cache);
        }

        pub inline fn zero(
            self: *const Self,
            pipeline: *Pipeline,
        ) anyerror!void {
            try self.vtable.zero(self.ptr, pipeline);
        }

        pub inline fn deinit(self: *const Self, pipeline: *Pipeline) void {
            self.vtable.deinit(self.ptr, pipeline);
        }
    };
}

test {
    _ = gd_module;
    _ = gdm_module;
    _ = adagrad_module;
    _ = rmsprop_module;
}
