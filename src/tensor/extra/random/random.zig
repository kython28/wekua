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
const wTensorDtypes = dtypes.wTensorDtype;

const random_cl_kernel = @embedFile("../../../../kernels/tensor/extra/random.cl");

fn get_kernel(command_queue: wCommandQueue, dtype: wTensorDtype) !cl.kernel.cl_kernel {
    const kernel = try w_kernel.get_only_float_kernel(command_queue, .Random);

}

pub fn random(command_queue: wCommandQueue, tensor: wTensor) !void {
    const dtype = tensor.dtype;
    switch (dtype) {
        .float32,.float64 => {},
        else => return w_errors.errors.UnsupportedDataType
    }
}
