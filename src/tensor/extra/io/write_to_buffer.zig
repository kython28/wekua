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

const utils = @import("utils.zig");

pub fn write_to_buffer(command_queue: wCommandQueue, tensor: wTensor, buffer: anytype) !void {
    const dtype = tensor.dtype;
    try utils.check_buffer_type(dtype, buffer);

    const dtype_size = dtypes.get_dtype_size(dtype);
    const number_of_elements = tensor.number_of_elements;

    const tensor_shape = tensor.shape;
    const c: usize = @intCast(
        tensor_shape[tensor_shape.len - 1] * (
            1 + @as(u64, @intCast(@intFromBool(tensor.is_complex)))
        )
    );
    const r: usize = number_of_elements / tensor.row_pitch;

    const buff_origin: [3]usize = .{0, 0, 0};
    const region: [3]usize = .{c*dtype_size, r, 1};
    const buf_row_pitch = tensor.row_pitch * dtype_size;
    const host_row_pitch = c * dtype_size;

    const prev_events = w_event.acquire_tensor(tensor, .read);
    const tensor_mutex = &tensor.mutex;
    errdefer tensor_mutex.unlock();

    const buffer_as_bytes = std.mem.asBytes(buffer);

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.read_rect(
        command_queue.cmd, tensor.buffer, false, &buff_origin, &buff_origin, &region,
        buf_row_pitch, 0, host_row_pitch, 0, buffer_as_bytes.ptr, prev_events, &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    var cond = std.Thread.Condition{};
    try w_event.register_new_event(
        command_queue, tensor, &utils.signal_condition_callback, &cond, new_event, .read
    );
    cond.wait(tensor_mutex);
    tensor_mutex.unlock();
}
