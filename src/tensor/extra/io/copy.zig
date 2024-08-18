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

const validations = @import("../../utils/validations.zig");

fn copy_tensor_with_different_row_pitch(command_queue: wCommandQueue, src: wTensor, dst: wTensor) !void {

}

fn copy_tensor_with_same_row_pitch(command_queue: wCommandQueue, src: wTensor, dst: wTensor) !void {
    const src_prev_events = w_event.acquire_tensor(src, .read);
    defer src.mutex.unlock();

    const dst_prev_events = w_event.acquire_tensor(dst, .write);
    defer dst.mutex.unlock();



}

pub fn copy(command_queue: wCommandQueue, src: wTensor, dst: wTensor) !void {
    try validations.eql_tensors_dimensions(src, dst);

    if (src.row_pitch == dst.row_pitch) {
        try copy_tensor_with_same_row_pitch(command_queue, src, dst);
    }else{
        try copy_tensor_with_different_row_pitch(command_queue, src, dst);
    }
}
