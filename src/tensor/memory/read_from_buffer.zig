const cl = @import("opencl");

const core = @import("../../core/main.zig");
const CommandQueue = core.CommandQueue;

const helpers = @import("../helpers.zig");

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;


pub fn readFromBuffer(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
    buffer: []const T,
) !void {
    if (buffer.len != tensor.number_of_elements_without_padding) {
        return w_tensor.Errors.InvalidBuffer;
    }

    const tensor_shape = tensor.shape;
    const c: usize = @intCast(
        tensor_shape[tensor_shape.len - 1] * (1 + @as(u64, @intCast(@intFromBool(tensor.is_complex)))),
    );
    const r: usize = tensor.number_of_elements / tensor.row_pitch;

    const buff_origin: [3]usize = .{ 0, 0, 0 };
    const region: [3]usize = .{ c * @sizeOf(T), r, 1 };
    const buf_row_pitch = tensor.row_pitch * @sizeOf(T);
    const host_row_pitch = c * @sizeOf(T);

    const prev_events = tensor.events_manager.getPrevEvents(.read);

    var new_event: cl.event.cl_event = undefined;
    {
        try cl.buffer.write_rect(
            command_queue.cmd,
            tensor.buffer,
            false,
            &buff_origin,
            &buff_origin,
            &region,
            buf_row_pitch,
            0,
            host_row_pitch,
            0,
            buffer.ptr,
            prev_events,
            &new_event,
        );
        errdefer |err| helpers.release_event(new_event, err);

        try tensor.events_manager.appendNewEvent(.read, prev_events, new_event, null);
    }

    try tensor.wait();
}
