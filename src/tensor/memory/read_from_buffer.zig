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
    if (buffer.len != tensor.dimensions.number_of_elements_without_padding) {
        return w_tensor.Errors.InvalidBuffer;
    }

    const tensor_shape = tensor.dimensions.shape;
    const ndim = tensor_shape.len;
    const width: usize = (
        tensor_shape[ndim - 1] * (1 + @as(u64, @intFromBool(tensor.flags.is_complex)))
    ) * @sizeOf(T);
    const height: usize = if (ndim >= 2) tensor_shape[ndim - 2] else 1;

    var depth: usize = 1;
    if (ndim >= 3) {
        for (tensor_shape[0..ndim - 2]) |e| depth *= e;
    }

    const buff_origin: [3]usize = .{ 0, 0, 0 };
    const region: [3]usize = .{ width, height, depth };

    const buf_row_pitch = tensor.memory_layout.row_pitch * @sizeOf(T);
    const buf_slice_pitch = tensor.memory_layout.slice_pitch * @sizeOf(T);

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

        _ = try tensor.events_manager.appendNewEvent(.read, prev_events, new_event, null);
    }

    try tensor.wait();
}
