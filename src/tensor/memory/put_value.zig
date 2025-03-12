const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("../../core/main.zig").CommandQueue;

const w_event = @import("../utils/event.zig");
const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const utils = @import("utils.zig");

pub fn putValue(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
    coor: []const u64,
    real_scalar: ?T,
    imag_scalar: ?T,
) !void {
    const is_complex = tensor.is_complex;

    if (coor.len != tensor.shape.len) {
        return w_tensor.Errors.InvalidCoordinates;
    } else if (real_scalar == null and imag_scalar == null) {
        return w_tensor.Errors.InvalidValue;
    } else if (imag_scalar != null and !is_complex) {
        return w_tensor.Errors.TensorDoesNotSupportComplexNumbers;
    }

    const allocator = command_queue.allocator;

    const Resources = packed struct {
        size: usize,
        pattern: [2]T,
    };

    const resources = try allocator.create(Resources);
    errdefer allocator.free(resources);

    resources.size = @sizeOf(Resources);

    const pattern: []T = &resources.pattern;

    pattern[0] = real_scalar orelse 0;
    pattern[1] = imag_scalar orelse 0;

    var stride: usize = @intCast(tensor.number_of_elements);
    var offset: usize = 0;

    const shape = tensor.shape;
    const last_index = shape.len - 1;
    for (shape[0..last_index], coor[0..last_index]) |v, c| {
        stride /= v;
        offset += c * stride;
    }
    offset += coor[last_index] * (1 + @as(usize, @intCast(@intFromBool(is_complex))));
    offset *= @sizeOf(T);

    const prev_events = w_event.acquire_tensor(tensor, .write);
    defer tensor.mutex.unlock();

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.write(
        command_queue.cmd,
        tensor.buffer,
        false,
        offset,
        @sizeOf(T) * (1 + @as(usize, is_complex)),
        pattern.ptr,
        prev_events,
        &new_event,
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    try w_event.register_new_event_to_single_tensor(
        command_queue,
        tensor,
        &utils.release_temporal_resource_callback,
        resources,
        new_event,
        .write,
    );
}
