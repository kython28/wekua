const cl = @import("opencl");

const core = @import("../../core/main.zig");
const CommandQueue = core.CommandQueue;

const helpers = @import("../helpers.zig");

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

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
    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ src_prev_events, dst_prev_events },
        null,
    );
    errdefer events_set.release();
    const prev_events = events_set.getPrevEvents();

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
    errdefer |err| helpers.releaseEvent(new_event, err);

    try events_set.appendNewEvent(T, &.{ .read, .write }, &.{ src, dst }, prev_events, new_event);
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
    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ src_prev_events, dst_prev_events },
        null,
    );
    errdefer events_set.release();
    const prev_events = events_set.getPrevEvents();

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
    errdefer |err| helpers.releaseEvent(new_event, err);

    try events_set.appendNewEvent(T, &.{ .read, .write }, &.{ src, dst }, prev_events, new_event);
}

pub fn copy(comptime T: type, command_queue: *const CommandQueue, src: *Tensor(T), dst: *Tensor(T)) !void {
    try helpers.eqlTensorsDimensions(T, src, dst);
    try helpers.eqlNumberSpace(T, src, dst);

    if (src.row_pitch == dst.row_pitch) {
        try copy_tensor_with_same_row_pitch(T, command_queue, src, dst);
    } else {
        try copy_tensor_with_different_row_pitch(T, command_queue, src, dst);
    }
}
