const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;

const helpers = @import("../helpers.zig");

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;


pub fn writeToBuffer(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    buffer: []T,
) TensorErrors!void {
    if (buffer.len != tensor.dimensions.number_of_elements_without_padding) {
        return tensor_module.Errors.InvalidBuffer;
    }

    const tensor_shape = tensor.dimensions.shape;
    const ndim = tensor_shape.len;
    const width: usize = tensor_shape[ndim - 1] * @sizeOf(T);
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

    const prev_events = pipeline.prevEvents();

    var new_event: cl.event.Event = undefined;
    try cl.buffer.readRect(
        pipeline.command_queue.cl_command_queue,
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
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}
