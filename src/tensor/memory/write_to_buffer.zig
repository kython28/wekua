const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("../../core/main.zig").CommandQueue;

const w_event = @import("../utils/event.zig");
const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const utils = @import("utils.zig");

pub fn writeToBuffer(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
    buffer: []T,
) !void {
    const number_of_elements = tensor.number_of_elements;

    const tensor_shape = tensor.shape;
    const c: usize = @intCast(
        tensor_shape[tensor_shape.len - 1] * (1 + @as(u64, @intCast(@intFromBool(tensor.is_complex)))),
    );
    const r: usize = number_of_elements / tensor.row_pitch;

    const buff_origin: [3]usize = .{ 0, 0, 0 };
    const region: [3]usize = .{ c * @sizeOf(T), r, 1 };
    const buf_row_pitch = tensor.row_pitch * @sizeOf(T);
    const host_row_pitch = c * @sizeOf(T);

    const prev_events = w_event.acquire_tensor(tensor, .read);
    const tensor_mutex = &tensor.mutex;
    errdefer tensor_mutex.unlock();

    const buffer_as_bytes = @as([*]T, @ptrCast(buffer.ptr))[0..(buffer.len * @sizeOf(T))];

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.read_rect(
        command_queue.cmd,
        tensor.buffer,
        false,
        &buff_origin,
        &buff_origin,
        &region,
        buf_row_pitch,
        0,
        host_row_pitch,
        0,
        buffer_as_bytes.ptr,
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
}
