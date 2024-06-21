const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_empty = @import("../empty.zig");
const w_event = @import("../utils/event.zig");
const w_errors = @import("../utils/errors.zig");

const dtypes = @import("../utils/dtypes.zig");
const wTensor = dtypes.wTensor;

const random_cl_kernel = @embedFile("kernels/tensor/extra/random.cl");
// const randn_cl_kernel = @embedFile("kernels/tensor/extra/randn.cl");

pub fn random(tensor: wTensor) !void {
    const dtype = tensor.dtype;
    switch (tensor.d)
}
