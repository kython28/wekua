const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_event = @import("../../utils/event.zig");
const w_errors = @import("../../utils/errors.zig").errors;
const w_kernel = @import("../../../core/kernel.zig");

const dtypes = @import("../../utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wTensorDtype = dtypes.wTensorDtype;
const wScalar = dtypes.wScalar;

const validations = @import("../../utils/validations.zig");

const to_complex_cl_kernel: []const u8 = @embedFile("kernels/to_complex.cl");

const to_complex_resources = struct {
    prev_events: []cl.event.cl_event
};

fn release_events_array(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const resources: *to_complex_resources = @ptrCast(@alignCast(data));
        allocator.free(resources.prev_events);
        allocator.destroy(resources);
    }
}

inline fn get_kernel_index(_: bool, _: bool, dtype: wTensorDtype, dom: bool) usize {
    return 2 * @as(usize, @intFromEnum(dtype)) + @intFromBool(dom);
}

pub fn to_complex(
    command_queue: wCommandQueue, src: wTensor, dst: wTensor, dom: bool
) !void {
    try validations.eql_tensors_dimensions(src, dst);
    if (src.is_complex) return w_errors.InvalidValue;
    if (!dst.is_complex) return w_errors.InvalidValue;


    const allocator = command_queue.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(
        allocator, "-DOFFSET={d}", .{@intFromBool(dom)}
    );
    defer allocator.free(extra_args);
    
    const kernel = try w_kernel.create_and_get_kernel(
        command_queue, .ToComplex, to_complex_cl_kernel, .{
            .dtype = src.dtype,
            .is_complex = false,
            .vectors_enabled = false,
            .kernel_name = "to_complex",
            .extra_args = extra_args
        }, false, false, dtypes.number_of_dtypes * 2, get_kernel_index, dom
    );

    const cmd = command_queue.cmd;

    const src_prev_events = w_event.acquire_tensor(src, .read);
    defer src.mutex.unlock();

    const dst_prev_events = w_event.acquire_tensor(dst, .write);
    defer dst.mutex.unlock();

    const prev_events = try w_event.concatenate_events(allocator, &.{src_prev_events, dst_prev_events});
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&src.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&dst.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&src.row_pitch));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&dst.row_pitch));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd, kernel, null, &src.shape_like_matrix_without_vectors,
        &src.work_items_like_matrix_without_vectors[command_queue.wekua_id],
        null, &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    var resources: ?*to_complex_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(to_complex_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (resources) |v| allocator.destroy(v);
    }

    try w_event.register_new_event_to_multiple_tensors(
        command_queue, &.{src, dst}, &release_events_array, resources, new_event, &.{.read, .write}
    );
}

