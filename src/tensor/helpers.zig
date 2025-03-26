const std = @import("std");
const cl = @import("opencl");

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

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

pub fn releaseEvent(event: cl.event.cl_event, err: anyerror) void {
    cl.event.wait(event) catch |err2| {
        std.debug.panic(
            "An error ocurred ({s}) while waiting for new event and dealing with another error ({s})",
            .{
                @errorName(err2),
                @errorName(err),
            },
        );
    };
    cl.event.release(event);
}

pub inline fn eqlNumberSpace(comptime T: type, tensor_a: *Tensor(T), tensor_b: *Tensor(T)) !void {
    if (tensor_a.is_complex != tensor_b.is_complex) return w_tensor.Errors.TensorDoesNotSupportComplexNumbers;
}

pub inline fn eqlTensorsDimensions(comptime T: type, tensor_a: *Tensor(T), tensor_b: *Tensor(T)) !void {
    if (!std.mem.eql(u64, tensor_a.shape, tensor_b.shape)) return w_tensor.Errors.UnqualTensorsShape;
}

// pub inline fn eql_tensors(comptime T: type, tensor_a: *Tensor(T), tensor_b: *Tensor(T)) !void {
//     try eqlTensorsDimensions(T, tensor_a, tensor_b);
//     try eqlNumberSpace(T, tensor_a, tensor_b);
//     if (tensor_a.vectors_enabled != tensor_b.vectors_enabled) return w_tensor.Errors.UnqualTensorsAttribute;
// }
