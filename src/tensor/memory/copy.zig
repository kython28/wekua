const std = @import("std");
const cl = @import("opencl");

const core = @import("../../core/main.zig");
const CommandQueue = core.CommandQueue;

const helpers = @import("../helpers.zig");

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const validations = @import("../utils/validations.zig");

const CopyData = struct {
    allocator: std.mem.Allocator,
    prev_events: []cl.event.cl_event,
};

fn release_resources(ptr: ?*anyopaque) void {
    const maybe_data: ?*CopyData = @alignCast(@ptrCast(ptr));
    const data = maybe_data orelse return;

    const allocator = data.allocator;
    allocator.free(data.prev_events);
    allocator.destroy(data);
}

fn copy_tensor_with_different_row_pitch(
    comptime T: type,
    command_queue: *const CommandQueue,
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

    const src_prev_events = src.events_manager.getPrevEvents(.read);
    const dst_prev_events = dst.events_manager.getPrevEvents(.write);

    const allocator = command_queue.allocator;
    const prev_events = try w_tensor.EventManager.concat(allocator, &.{ src_prev_events, dst_prev_events });
    errdefer {
        if (prev_events) |v| {
            allocator.free(v);
        }
    }

    var copy_data: ?*CopyData = null;
    if (prev_events) |pv| {
        const ptr = try allocator.create(CopyData);
        errdefer allocator.destroy(ptr);

        ptr.* = .{
            .allocator = allocator,
            .prev_events = pv,
        };

        copy_data = ptr;
    }
    errdefer {
        if (copy_data) |ptr| {
            allocator.destroy(ptr);
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
    errdefer |err| helpers.release_event(new_event, err);

    try w_tensor.EventManager.appendNewEventToMultipleTensor(
        T,
        allocator,
        &.{ .read, .write },
        &.{ src, dst },
        prev_events,
        new_event,
        .{ .data = copy_data, .func = &release_resources },
    );
}

fn copy_tensor_with_same_row_pitch(
    comptime T: type,
    command_queue: *const CommandQueue,
    src: *Tensor(T),
    dst: *Tensor(T),
) !void {
    const src_prev_events = src.events_manager.getPrevEvents(.read);
    const dst_prev_events = dst.events_manager.getPrevEvents(.write);

    const allocator = command_queue.allocator;
    const prev_events = try w_tensor.EventManager.concat(allocator, &.{ src_prev_events, dst_prev_events });
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    var copy_data: ?*CopyData = null;
    if (prev_events) |pv| {
        const ptr = try allocator.create(CopyData);
        errdefer allocator.destroy(ptr);

        ptr.* = .{
            .allocator = allocator,
            .prev_events = pv,
        };

        copy_data = ptr;
    }
    errdefer {
        if (copy_data) |ptr| {
            allocator.destroy(ptr);
        }
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

    errdefer |err| helpers.release_event(new_event, err);

    try w_tensor.EventManager.appendNewEventToMultipleTensor(
        T,
        allocator,
        &.{ .read, .write },
        &.{ src, dst },
        prev_events,
        new_event,
        .{ .data = copy_data, .func = &release_resources },
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
