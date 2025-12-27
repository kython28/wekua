const builtin = @import("builtin");
const std = @import("std");

const cl = @import("opencl");

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;
const Errors = tensor_module.Errors;

pub const prevEventsResource = struct {
    allocator: std.mem.Allocator,
    prev_events: []cl.event.cl_event,
};

pub fn releaseEventsArray(user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const resources: *prevEventsResource = @ptrCast(@alignCast(data));

        resources.allocator.free(resources.prev_events);
        resources.allocator.destroy(resources);
    }
}

pub fn createPrevEventsResource(
    allocator: std.mem.Allocator,
    prev_events: ?[]cl.event.cl_event,
) !?*prevEventsResource {
    const prev_events_array = prev_events orelse return null;

    const resources = try allocator.create(prevEventsResource);
    errdefer allocator.destroy(resources);

    resources.* = .{
        .allocator = allocator,
        .prev_events = prev_events_array,
    };

    return resources;
}

pub fn releaseEvent(event: cl.event.Event) void {
    cl.event.wait(event) catch |err| {
        std.debug.panic(
            "An error ocurred ({s}) while waiting for new event and dealing with another error",
            .{
                @errorName(err),
            },
        );
    };
    cl.event.release(event);
}

pub inline fn eqlTensorsShape(comptime T: type, tensor_a: *Tensor(T), tensor_b: *Tensor(T)) Errors!void {
    if (!std.mem.eql(u64, tensor_a.dimensions.shape, tensor_b.dimensions.shape)) {
        return Errors.UnqualTensorsShape;
    }
}

pub inline fn eqlTensors(comptime T: type, tensor_a: *Tensor(T), tensor_b: *Tensor(T)) Errors!void {
    try eqlTensorsShape(T, tensor_a, tensor_b);

    if (tensor_a.flags.vectors_enabled != tensor_b.flags.vectors_enabled) {
        return Errors.UnqualTensorsAttribute;
    }
}
