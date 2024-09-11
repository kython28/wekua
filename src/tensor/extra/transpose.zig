const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_kernel = @import("../../core/kernel.zig");

const w_event = @import("../utils/event.zig");
const w_errors = @import("../utils/errors.zig").errors;
const w_copy = @import("io/copy.zig");

const dtypes = @import("../utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wTensorDtype = dtypes.wTensorDtype;

const validations = @import("../utils/validations.zig");

const transpose_cl_kernel: []const u8 = @embedFile("kernels/transpose.cl");

const transpose_resources =  struct {
    prev_events: []cl.event.cl_event
};

pub fn release_events_array(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const resources: *transpose_resources = @ptrCast(@alignCast(data));
        allocator.free(resources.prev_events);
        allocator.destroy(resources);
    }
}

pub fn transpose(command_queue: wCommandQueue, result_tensor: wTensor, tensor: wTensor, dim0: u64, dim1: u64) !void {
    try validations.eql_tensors_dtype(tensor, result_tensor);
    try validations.eql_number_space(tensor, result_tensor);

    const shape_a = result_tensor.shape;
    const shape_b = tensor.shape;
    if (shape_a.len != shape_b.len) return w_errors.UnqualTensorsDimension;
    if (dim0 >= shape_a.len or dim1 >= shape_a.len) return w_errors.InvalidValue;
    if (tensor.number_of_elements_without_padding != result_tensor.number_of_elements_without_padding) {
        return w_errors.UnqualTensorsDimension;
    }
    if (shape_a[dim0] != shape_b[dim1] or shape_a[dim1] != shape_b[dim0]) return w_errors.InvalidValue;
    if (dim0 == dim1) {
        try w_copy.copy(command_queue, tensor, result_tensor);
        return;
    }

    const kernel = try w_kernel.get_cl_no_vector_kernel(
        command_queue, tensor, .Transpose, "transpose", transpose_cl_kernel, null
    );
    const cmd = command_queue.cmd;

    const src_prev_events = w_event.acquire_tensor(tensor, .read);
    defer tensor.mutex.unlock();

    const dst_prev_events = w_event.acquire_tensor(result_tensor, .write);
    defer result_tensor.mutex.unlock();

    const allocator = command_queue.allocator;
    const prev_events = try w_event.concatenate_events(allocator, &.{src_prev_events, dst_prev_events});
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const set_arg = cl.kernel.set_arg;
    const u64_size = @sizeOf(u64);
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);
    const shape = tensor.shape;
    const ndim: u64 = @intCast(shape.len);
    var dim0_: u64 = undefined;
    var dim1_: u64 = undefined;

    if (dim0 > dim1) {
        dim0_ = dim1;
        dim1_ = dim0;
    }else{
        dim0_ = dim0;
        dim1_ = dim1;
    }

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&tensor.pitchs_buffer));
    try set_arg(kernel, 2, cl_mem_size, @ptrCast(&result_tensor.buffer));
    try set_arg(kernel, 3, cl_mem_size, @ptrCast(&result_tensor.pitchs_buffer));
    try set_arg(kernel, 4, u64_size, @ptrCast(&dim0_));
    try set_arg(kernel, 5, u64_size, @ptrCast(&dim1_));
    try set_arg(kernel, 6, u64_size, @ptrCast(&ndim));

    // TODO: Adapt code to use views
    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd, kernel, null, &[1]u64{tensor.number_of_elements},
        tensor.work_item_for_all_elements[command_queue.wekua_id..command_queue.wekua_id+1],
        prev_events, &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    try cl.event.retain(new_event);
    errdefer cl.event.release(new_event) catch unreachable;

    var resources: ?*transpose_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(transpose_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (resources) |v| allocator.destroy(v);
    }

    try w_event.register_new_event_to_multiple_tensors(
        command_queue, &.{tensor, result_tensor}, &release_events_array, resources, new_event, &.{.read, .write}
    );
}
