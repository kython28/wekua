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

const copy_resources =  struct {
    prev_events: []cl.event.cl_event
};

pub fn release_events_array(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const resources: *copy_resources = @ptrCast(@alignCast(data));
        allocator.free(resources.prev_events);
        allocator.destroy(resources);
    }
}

fn copy_tensor_with_different_row_pitch(command_queue: wCommandQueue, src: wTensor, dst: wTensor) !void {
    const dtype = src.dtype;

    const dtype_size = dtypes.get_dtype_size(dtype);
    const number_of_elements = src.number_of_elements;

    const tensor_shape = src.shape;
    const c: usize = @intCast(
        tensor_shape[tensor_shape.len - 1] * (
            1 + @as(u64, @intCast(@intFromBool(src.is_complex)))
        )
    );
    const r: usize = number_of_elements / c;

    const buff_origin: [3]usize = .{0, 0, 0};
    const region: [3]usize = .{c*dtype_size, r, 1};
    const src_row_pitch = src.row_pitch * dtype_size;
    const dst_row_pitch = dst.row_pitch * dtype_size;

    const src_prev_events = w_event.acquire_tensor(src, .read);
    defer src.mutex.unlock();

    const dst_prev_events = w_event.acquire_tensor(dst, .write);
    defer dst.mutex.unlock();

    const allocator = command_queue.allocator;
    const prev_events = try w_event.concatenate_events(allocator, &.{src_prev_events, dst_prev_events});
    var not_registered: bool = true;
    errdefer {
        if (not_registered) {
            if (prev_events) |v| {
                allocator.free(v);
            }
        }
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.copy_rect(
        command_queue.cmd, src.buffer, dst.buffer, &buff_origin, &buff_origin, &region, src_row_pitch, 0,
        dst_row_pitch, 0, prev_events, &new_event
    );
    errdefer {
        if (not_registered) {
            cl.event.wait(new_event) catch unreachable;
            cl.event.release(new_event) catch unreachable;
        }
    }

    try cl.event.retain(new_event);
    errdefer cl.event.release(new_event) catch unreachable;

    var resources: ?*copy_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(copy_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (not_registered) {
            if (resources) |v| allocator.destroy(v);
        }
    }

    try w_event.register_new_event(command_queue, src, &release_events_array, resources, new_event, .read);
    not_registered = false;
    try w_event.register_new_event(command_queue, dst, null, resources, new_event, .write);
}

fn copy_tensor_with_same_row_pitch(command_queue: wCommandQueue, src: wTensor, dst: wTensor) !void {
    const src_prev_events = w_event.acquire_tensor(src, .read);
    defer src.mutex.unlock();

    const dst_prev_events = w_event.acquire_tensor(dst, .write);
    defer dst.mutex.unlock();

    const allocator = command_queue.allocator;
    const prev_events = try w_event.concatenate_events(allocator, &.{src_prev_events, dst_prev_events});
    var not_registered: bool = true;
    errdefer {
        if (not_registered) {
            if (prev_events) |v| allocator.free(v);
        }
    }

    const size = src.size;
    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.copy(
        command_queue.cmd, src.buffer, dst.buffer, 0, 0, size, prev_events, &new_event
    );

    errdefer {
        if (not_registered) {
            cl.event.wait(new_event) catch unreachable;
            cl.event.release(new_event) catch unreachable;
        }
    }

    try cl.event.retain(new_event);
    errdefer cl.event.release(new_event) catch unreachable;

    var resources: ?*copy_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(copy_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (not_registered) {
            if (resources) |v| allocator.destroy(v);
        }
    }

    try w_event.register_new_event(command_queue, src, &release_events_array, resources, new_event, .read);
    not_registered = true;
    try w_event.register_new_event(command_queue, dst, null, resources, new_event, .write);
}

pub fn copy(command_queue: wCommandQueue, src: wTensor, dst: wTensor) !void {
    try validations.eql_tensors_dimensions(src, dst);

    if (src.row_pitch == dst.row_pitch) {
        try copy_tensor_with_same_row_pitch(command_queue, src, dst);
    }else{
        try copy_tensor_with_different_row_pitch(command_queue, src, dst);
    }
}
