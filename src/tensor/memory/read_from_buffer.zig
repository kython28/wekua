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
    const width: usize = (
        tensor_shape[tensor_shape.len - 1] * (1 + @as(u64, @intFromBool(tensor.is_complex)))
    ) * @sizeOf(T);
    const height: usize = tensor_shape[tensor_shape.len - 2];

    var depth: usize = 1;
    if (tensor_shape.len >= 3) {
        for (tensor_shape[0..tensor_shape.len - 2]) |e| depth *= e;
    }

    const buff_origin: [3]usize = .{ 0, 0, 0 };
    const region: [3]usize = .{ width, height, depth };

    const buf_row_pitch = tensor.row_pitch * @sizeOf(T);
    const buf_slice_pitch = tensor.slice_pitch * @sizeOf(T);

    const host_row_pitch = width;
    const host_slice_pitch = height * host_row_pitch;

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
            buf_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            buffer.ptr,
            prev_events,
            &new_event,
        );
        errdefer |err| helpers.releaseEvent(new_event, err);

        try tensor.events_manager.appendNewEvent(.read, prev_events, new_event, null);
    }

    try tensor.wait();
}
