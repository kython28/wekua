const std = @import("std");
const cl = @import("opencl");

const wCommandQueue = @import("../../../core/command_queue.zig");

const w_kernel = @import("../../../core/kernel.zig");

const w_event = @import("../../utils/event.zig");
const w_tensor = @import("../../main.zig");
const wTensor = w_tensor.wTensor;

const random_cl_kernel: []const u8 = @embedFile("kernels/random.cl");

fn fill_with_random_bytes(
    cmd: cl.command_queue.cl_command_queue,
    tensor_buf: cl.buffer.cl_mem,
    size: usize,
    prev_events: ?[]const cl.event.cl_event,
) !void {
    var event: cl.event.cl_event = undefined;
    const map_flags = @intFromEnum(cl.buffer.enums.map_flags.read) | @intFromEnum(cl.buffer.enums.map_flags.write);
    const tensor_map: []u8 = try cl.buffer.map([]u8, cmd, tensor_buf, false, map_flags, 0, size, prev_events, &event);

    cl.event.wait(event) catch unreachable;
    cl.event.release(event) catch unreachable;
    defer cl.buffer.unmap([]u8, cmd, tensor_buf, tensor_map, null, null) catch unreachable;

    std.crypto.random.bytes(tensor_map);

    try cl.buffer.write(cmd, tensor_buf, false, 0, size, tensor_map.ptr, null, &event);
    cl.event.wait(event) catch unreachable;
    cl.event.release(event) catch unreachable;
}

pub fn random(
    comptime T: type,
    tensor: *wTensor(T),
    command_queue: wCommandQueue,
) !void {
    const kernel = try w_kernel.get_cl_no_vector_kernel(
        command_queue,
        tensor,
        .Random,
        "random",
        random_cl_kernel,
        null,
    );
    const cmd = command_queue.cmd;
    const tensor_buf = &tensor.buffer;

    const prev_events = w_event.acquire_tensor(tensor, .write);
    defer tensor.mutex.unlock();

    try fill_with_random_bytes(cmd, tensor_buf.*, tensor.size, prev_events);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);
    const shape = tensor.shape;

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(tensor_buf));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(tensor_buf));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.row_pitch));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&shape[shape.len - 1]));

    // TODO: Adapt code to use views
    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &tensor.shape_like_matrix_without_vectors,
        &tensor.work_items_like_matrix_without_vectors[command_queue.wekua_id],
        null,
        &new_event,
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    try w_event.register_new_event_to_single_tensor(command_queue, tensor, null, null, new_event, .write);
}
