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

const random_cl_kernel = @embedFile("kernels/random.cl");

fn get_kernel(command_queue: wCommandQueue, tensor: wTensor) !cl.kernel.cl_kernel {
    const dtype = tensor.dtype;
    const is_complex = tensor.is_complex;
    const vectors_enabled = tensor.vectors_enabled;

    // It is 8 because: float32, float64, complex_float32, complex_float64 and the same without vectors
    const kernels_set = try w_kernel.get_kernel(command_queue, .Random, 8);

    var index: usize = (@intFromBool(vectors_enabled) * @as(usize, 4) +
        @intFromBool(is_complex) * @as(usize, 2));
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
            .vectors_enabled = tensor.vectors_enabled,
            .kernel_name = "random"
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
    size: usize, prev_events: []cl.event.cl_event
) !void {
    var mapping_event: cl.event.cl_event = undefined;
    const map_flags = @intFromEnum(cl.buffer.enums.map_flags.read)|@intFromEnum(cl.buffer.enums.map_flags.write);
    const tensor_map: []u8 = try cl.buffer.map(
        []u8, cmd, tensor_buf, false, map_flags, 0, size, prev_events, &mapping_event
    );

    cl.event.wait(mapping_event) catch unreachable;
    cl.event.release(mapping_event) catch unreachable;

    std.crypto.random.bytes(tensor_map);

    try cl.buffer.unmap([]u8, cmd, tensor_buf, tensor_map, null, null);
}

pub fn random(command_queue: wCommandQueue, tensor: wTensor) !void {
    const dtype = tensor.dtype;
    switch (dtype) {
        .float32,.float64 => {},
        else => return w_errors.errors.UnsupportedDataType
    }

    _ = try get_kernel(command_queue, tensor);
    // const kernel = try get_kernel(command_queue, tensor);
    // const cmd = command_queue.cmd;
    // const tensor_buf = tensor.buffer;

    // const prev_events = w_event.acquire_tensor(tensor, .write);
    // defer tensor.mutex.unlock();

    // try fill_with_random_bytes(cmd, tensor_buf, tensor.size, prev_events);

    // const set_arg = cl.kernel.set_arg;

    // try set_arg(kernel, 0, )
}
