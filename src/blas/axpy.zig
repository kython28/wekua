const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_event = @import("../tensor/utils/event.zig");
const w_errors = @import("../tensor/utils/errors.zig").errors;
const w_kernel = @import("../core/kernel.zig");

const dtypes = @import("../tensor/utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wTensorDtype = dtypes.wTensorDtype;
const wScalar = dtypes.wScalar;

const validations = @import("../tensor/utils/validations.zig");

const axpy_cl_kernel: []const u8 = @embedFile("kernels/axpy.cl");

const axpy_resources = struct {
    prev_events: []cl.event.cl_event
};

fn release_events_array(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const resources: *axpy_resources = @ptrCast(@alignCast(data));
        allocator.free(resources.prev_events);
        allocator.destroy(resources);
    }
}

pub fn axpy(command_queue: wCommandQueue, x: wTensor, alpha: wScalar, y: wTensor) !void {
    try validations.eql_number_space(x, y);
    try validations.eql_tensors_dimensions(x, y);

    const dtype = x.dtype;
    if (dtype != @as(wTensorDtype, alpha)) {
        return w_errors.InvalidScalarDtype;
    }

    const allocator = command_queue.allocator;
    const kernel = try w_kernel.get_cl_no_vector_no_complex_single_kernel_per_dtype(
        command_queue, x, .AXPY, "axpy", axpy_cl_kernel, null
    );

    const cmd = command_queue.cmd;

    const x_prev_events = w_event.acquire_tensor(x, .read);
    defer x.mutex.unlock();

    const y_prev_events = w_event.acquire_tensor(y, .write);
    defer y.mutex.unlock();

    const prev_events = try w_event.concatenate_events(allocator, &.{x_prev_events, y_prev_events});
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);
    const alpha_dtype = dtypes.get_dtype_size(dtype);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, alpha_dtype, &alpha);

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd, kernel, null, @as([*]const u64, @ptrCast(&x.number_of_vectors))[0..1],
        @as([*]const u64, @ptrCast(&x.work_item_for_all_vectors[command_queue.wekua_id]))[0..1],
        null, &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    var resources: ?*axpy_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(axpy_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (resources) |v| allocator.destroy(v);
    }

    try w_event.register_new_event_to_multiple_tensors(
        command_queue, &.{x, y}, &release_events_array, resources, new_event, &.{.read, .write}
    );
}

