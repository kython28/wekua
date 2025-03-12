const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("../../core/main.zig").CommandQueue;

const w_event = @import("../utils/event.zig");
const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const utils = @import("utils.zig");

pub fn getValue(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
    coor: []const u64,
    real_scalar: ?*T,
    imag_scalar: ?*T,
) !void {
    const is_complex = tensor.is_complex;
    if (coor.len != tensor.shape.len) {
        return w_tensor.wTensorErrors.InvalidCoordinates;
    } else if (real_scalar == null and imag_scalar == null) {
        return w_tensor.wTensorErrors.InvalidValue;
    } else if (imag_scalar != null and !is_complex) {
        return w_tensor.wTensorErrors.TensorDoesNotSupportComplexNumbers;
    }

    var buf_size: usize = @sizeOf(T);
    if (is_complex) {
        buf_size += @sizeOf(T);
    }

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

    const prev_events = w_event.acquire_tensor(tensor, .read);
    const tensor_mutex = &tensor.mutex;
    errdefer tensor_mutex.unlock();

    var new_event: cl.event.cl_event = undefined;
    var buf: [2]T = undefined;

    try cl.buffer.read(
        command_queue.cmd,
        tensor.buffer,
        false,
        offset,
        buf_size,
        &buf,
        prev_events,
        &new_event,
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    var cond = std.Thread.Condition{};
    try w_event.register_new_event_to_single_tensor(
        command_queue,
        tensor,
        &utils.signal_condition_callback,
        &cond,
        new_event,
        .read,
    );
    cond.wait(tensor_mutex);
    tensor_mutex.unlock();

    if (real_scalar) |scalar| {
        scalar.* = buf[0];
    }

    if (imag_scalar) |scalar| {
        scalar.* = buf[1];
    }
}
