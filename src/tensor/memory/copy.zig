const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("../../core/main.zig").CommandQueue;

const w_event = @import("../utils/event.zig");
const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const utils = @import("utils.zig");

const validations = @import("../utils/validations.zig");

const copy_resources = struct { prev_events: []cl.event.cl_event };

fn release_events_array(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const resources: *copy_resources = @ptrCast(@alignCast(data));
        allocator.free(resources.prev_events);
        allocator.destroy(resources);
    }
}

fn copy_tensor_with_different_row_pitch(
    comptime T: type,
    command_queue: CommandQueue,
    src: *Tensor(T),
    dst: *Tensor(T),
) !void {
    const number_of_elements = src.number_of_elements;

    const tensor_shape = src.shape;
    const c: usize = @intCast(
        tensor_shape[tensor_shape.len - 1] * (1 + @as(u64, @intCast(@intFromBool(src.is_complex)))),
    );
    const r: usize = number_of_elements / src.row_pitch;

    const buff_origin: [3]usize = .{ 0, 0, 0 };
    const region: [3]usize = .{ c * @sizeOf(T), r, 1 };
    const src_row_pitch = src.row_pitch * @sizeOf(T);
    const dst_row_pitch = dst.row_pitch * @sizeOf(T);

    const src_prev_events = w_event.acquire_tensor(src, .read);
    defer src.mutex.unlock();

    const dst_prev_events = w_event.acquire_tensor(dst, .write);
    defer dst.mutex.unlock();

    const allocator = command_queue.allocator;
    const prev_events = try w_event.concatenate_events(
        allocator,
        &.{
            src_prev_events,
            dst_prev_events,
        },
    );
    errdefer {
        if (prev_events) |v| {
            allocator.free(v);
        }
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.copy_rect(
        command_queue.cmd,
        src.buffer,
        dst.buffer,
        &buff_origin,
        &buff_origin,
        &region,
        src_row_pitch,
        0,
        dst_row_pitch,
        0,
        prev_events,
        &new_event,
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    try cl.event.retain(new_event);
    errdefer cl.event.release(new_event) catch unreachable;

    var resources: ?*copy_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(copy_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (resources) |v| allocator.destroy(v);
    }

    try w_event.register_new_event_to_multiple_tensors(
        command_queue,
        &.{ src, dst },
        &release_events_array,
        resources,
        new_event,
        &.{ .read, .write },
    );
}

fn copy_tensor_with_same_row_pitch(
    comptime T: type,
    command_queue: *const CommandQueue,
    src: *Tensor(T),
    dst: *Tensor(T),
) !void {
    const src_prev_events = w_event.acquire_tensor(src, .read);
    defer src.mutex.unlock();

    const dst_prev_events = w_event.acquire_tensor(dst, .write);
    defer dst.mutex.unlock();

    const allocator = command_queue.allocator;
    const prev_events = try w_event.concatenate_events(
        allocator,
        &.{ src_prev_events, dst_prev_events },
    );
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const size = src.size;
    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.copy(
        command_queue.cmd,
        src.buffer,
        dst.buffer,
        0,
        0,
        size,
        prev_events,
        &new_event,
    );

    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    try cl.event.retain(new_event);
    errdefer cl.event.release(new_event) catch unreachable;

    var resources: ?*copy_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(copy_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (resources) |v| allocator.destroy(v);
    }

    try w_event.register_new_event_to_multiple_tensors(
        command_queue,
        &.{ src, dst },
        &release_events_array,
        resources,
        new_event,
        &.{ .read, .write },
    );
}

pub fn copy(comptime T: type, command_queue: *const CommandQueue, src: *Tensor(T), dst: *Tensor(T)) !void {
    try validations.eql_tensors_dimensions(T, src, dst);
    try validations.eql_number_space(T, src, dst);

    if (src.row_pitch == dst.row_pitch) {
        try copy_tensor_with_same_row_pitch(T, command_queue, src, dst);
    } else {
        try copy_tensor_with_different_row_pitch(T, command_queue, src, dst);
    }
}
