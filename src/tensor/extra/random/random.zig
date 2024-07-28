const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_kernel = @import("../../../core/kernel.zig");

const w_empty = @import("../../empty.zig");
const w_event = @import("../../utils/event.zig");
const w_errors = @import("../../utils/errors.zig");

const dtypes = @import("../../utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wTensorDtype = dtypes.wTensorDtype;

const random_cl_kernel: []const u8 = @embedFile("kernels/random.cl");

fn get_kernel(command_queue: wCommandQueue, tensor: wTensor) !cl.kernel.cl_kernel {
    const dtype = tensor.dtype;
    const is_complex = tensor.is_complex;

    // It is 8 because: float32, float64, complex_float32, complex_float64
    const kernels_set = try w_kernel.get_kernel(command_queue, .Random, 4);

    var index: usize = @intFromBool(is_complex) * @as(usize, 2);
    index += switch (dtype) {
        .float32 => 0,
        .float64 => 1,
        else => unreachable
    };

    if (kernels_set.kernels.?[index]) |kernel| {
        return kernel;
    }

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    try w_kernel.compile_kernel(
        command_queue, .{
            .dtype = dtype,
            .is_complex = is_complex,
            .vectors_enabled = false,
            .kernel_name = "random",
            // .extra_args = null
        },
        &kernel, &program,
        random_cl_kernel
    );

    kernels_set.kernels.?[index] = kernel;
    kernels_set.programs.?[index] = program;

    return kernel;
}

fn fill_with_random_bytes(
    cmd: cl.command_queue.cl_command_queue, tensor_buf: cl.buffer.cl_mem,
    size: usize, prev_events: ?[]cl.event.cl_event
) !void {
    var event: cl.event.cl_event = undefined;
    const map_flags = @intFromEnum(cl.buffer.enums.map_flags.read)|@intFromEnum(cl.buffer.enums.map_flags.write);
    const tensor_map: []u8 = try cl.buffer.map(
        []u8, cmd, tensor_buf, false, map_flags, 0, size, prev_events, &event
    );

    cl.event.wait(event) catch unreachable;
    cl.event.release(event) catch unreachable;
    defer cl.buffer.unmap([]u8, cmd, tensor_buf, tensor_map, null, null) catch unreachable;

    std.crypto.random.bytes(tensor_map);

    try cl.buffer.write(
        cmd, tensor_buf, false, 0, size, tensor_map.ptr, null, &event
    );
    cl.event.wait(event) catch unreachable;
    cl.event.release(event) catch unreachable;
}

pub fn random(command_queue: wCommandQueue, tensor: wTensor) !void {
    const dtype = tensor.dtype;
    switch (dtype) {
        .float32,.float64 => {},
        else => return w_errors.errors.UnsupportedDataType
    }

    const kernel = try get_kernel(command_queue, tensor);
    const cmd = command_queue.cmd;
    const tensor_buf = &tensor.buffer;

    const prev_events = w_event.acquire_tensor(tensor, .write);
    defer tensor.mutex.unlock();

    try fill_with_random_bytes(cmd, tensor_buf.*, tensor.size, prev_events);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);
    const shape = tensor.shape;
    const row_pitch = tensor.row_pitch;
    const rows = tensor.number_of_elements / row_pitch;

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(tensor_buf));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(tensor_buf));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&row_pitch));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&shape[shape.len - 1]));

    // TODO: Adapt code to use views
    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd, kernel, null, &[2]u64{rows, row_pitch},
        &tensor.work_items_like_matrix_without_vectors[command_queue.wekua_id],
        null, &new_event
    );

    try w_event.register_new_event(command_queue, tensor, null, null, new_event, .write);
}
