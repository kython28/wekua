const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_event = @import("../../utils/event.zig");
const w_errors = @import("../../utils/errors.zig");

const dtypes = @import("../../utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wScalar = dtypes.wScalar;
const wTensorDtype = dtypes.wTensorDtype;

fn write_to_buffer_callback(_: *const std.mem.Allocator, user_data: ?*anyopaque) void {
    const cond: *std.Thread.Condition = @alignCast(@ptrCast(user_data.?));
    cond.signal();
}

pub fn write_to_buffer(command_queue: wCommandQueue, tensor: wTensor, buffer: []u8) !void {
    const dtype_size = dtypes.get_dtype_size(tensor.dtype);
    const number_of_elements = tensor.number_of_elements;
    const expected_size = number_of_elements * dtype_size;
    if (expected_size != buffer.len) {
        return w_errors.errors.InvalidBuffer;
    }

    const tensor_size = tensor.size;
    const tensor_shape = tensor.shape;
    const c: usize = @intCast(
        tensor_shape[tensor_shape.len - 1] * (
            1 + @as(u64, @intCast(@intFromBool(tensor.is_complex)))
        )
    );
    const r: usize = number_of_elements / c;

    const buff_origin: [3]usize = .{0, 0, 0};
    const region: [3]usize = .{c*dtype_size, r, 1};
    const buf_row_pitch = tensor.col_pitch * dtype_size;
    const host_row_pitch = c * dtype_size;

    const prev_events = w_event.acquire_tensor(tensor, .read);
    const tensor_mutex = &tensor.mutex;
    errdefer tensor_mutex.unlock();

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.read_rect(
        command_queue.cmd, tensor.buffer, false, &buff_origin, &buff_origin, &region,
        buf_row_pitch, tensor_size, host_row_pitch, 0, buffer.ptr, prev_events, &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    var cond = std.Thread.Condition{};
    try w_event.register_new_event(
        command_queue, tensor, &write_to_buffer_callback, &cond, new_event, .read
    );
    cond.wait(tensor_mutex);
    tensor_mutex.unlock();
}
