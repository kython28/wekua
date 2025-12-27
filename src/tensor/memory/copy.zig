const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;

const helpers = @import("../helpers.zig");

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;


fn copy_tensor_with_different_row_pitch(
    comptime T: type,
    pipeline: *Pipeline,
    src: *Tensor(T),
    dst: *Tensor(T),
) TensorErrors!void {
    const tensor_shape = src.dimensions.shape;
    const ndim = tensor_shape.len;
    const width: usize = tensor_shape[ndim - 1] * @sizeOf(T);
    const height: usize = if (ndim >= 2) tensor_shape[ndim - 2] else 1;

    var depth: usize = 1;
    if (ndim >= 3) {
        for (tensor_shape[0 .. ndim - 2]) |e| depth *= e;
    }

    const buff_origin: [3]usize = .{ 0, 0, 0 };
    const region: [3]usize = .{ width, height, depth };

    const src_row_pitch = src.memory_layout.row_pitch * @sizeOf(T);
    const src_slice_pitch = src.memory_layout.slice_pitch * @sizeOf(T);

    const dst_row_pitch = dst.memory_layout.row_pitch * @sizeOf(T);
    const dst_slice_pitch = dst.memory_layout.slice_pitch * @sizeOf(T);

    const prev_events = pipeline.prevEvents();

    var new_event: cl.event.Event = undefined;
    try cl.buffer.copyRect(
        pipeline.command_queue.cl_command_queue,
        src.buffer,
        dst.buffer,
        &buff_origin,
        &buff_origin,
        &region,
        src_row_pitch,
        src_slice_pitch,
        dst_row_pitch,
        dst_slice_pitch,
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

fn copy_tensor_with_same_row_pitch(
    comptime T: type,
    pipeline: *Pipeline,
    src: *Tensor(T),
    dst: *Tensor(T),
) TensorErrors!void {
    const prev_events = pipeline.prevEvents();

    const size = src.memory_layout.size;
    var new_event: cl.event.Event = undefined;
    try cl.buffer.copy(
        pipeline.command_queue.cl_command_queue,
        src.buffer,
        dst.buffer,
        0,
        0,
        size,
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

pub fn copy(
    comptime T: type,
    pipeline: *Pipeline,
    src: *Tensor(T),
    dst: *Tensor(T),
) TensorErrors!void {
    try helpers.eqlTensorsShape(T, src, dst);

    if (src.memory_layout.row_pitch == dst.memory_layout.row_pitch) {
        try copy_tensor_with_same_row_pitch(T, pipeline, src, dst);
    } else {
        try copy_tensor_with_different_row_pitch(T, pipeline, src, dst);
    }
}
