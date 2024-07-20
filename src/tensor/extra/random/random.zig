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

const random_cl_kernel = @embedFile("../../../../kernels/tensor/extra/random.cl");

fn get_kernel(command_queue: wCommandQueue, tensor: wTensor) !cl.kernel.cl_kernel {
    const dtype = tensor.dtype;
    const is_complex = tensor.is_complex;
    const vectors_enabled = tensor.vectors_enabled;

    // It is 8 because: float32, float64, complex_float32, complex_float64 and the same without vectors
    const kernels_set = try w_kernel.get_kernel(command_queue, .Random, 8);

    const index: usize = (@intFromBool(vectors_enabled) * @as(usize, 4) +
        @intFromBool(is_complex) * @as(usize, 2) + switch (dtype) {
        .float32 => 0,
        .float64 => 1,
        else => unreachable
    });

    if (kernels_set.kernels[index]) |kernel| {
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

    kernels_set.kernels[index] = kernel;
    kernels_set.programs[index] = program;

    return kernel;
}

pub fn random(command_queue: wCommandQueue, tensor: wTensor) !void {
    const dtype = tensor.dtype;
    switch (dtype) {
        .float32,.float64 => {},
        else => return w_errors.errors.UnsupportedDataType
    }

    const kernel = try get_kernel(command_queue, tensor);
}
