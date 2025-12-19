const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;

const helpers = @import("../helpers.zig");

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

pub fn getValue(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    coor: []const u64,
    scalar: *T,
) TensorErrors!void {
    if (coor.len != tensor.dimensions.shape.len) {
        return tensor_module.Errors.InvalidCoordinates;
    }

    var offset: usize = 0;
    for (tensor.dimensions.pitches, tensor.dimensions.shape, coor) |p, ds, c| {
        if (c >= ds) return tensor_module.Errors.InvalidCoordinates;

        offset += c * p;
    }
    offset *= core.types.getTypeSize(T);

    const prev_events = pipeline.prevEvents();

    var new_event: cl.event.Event = undefined;
    try cl.buffer.read(
        pipeline.command_queue.cl_command_queue,
        tensor.buffer,
        false,
        offset,
        @sizeOf(T),
        scalar,
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}
