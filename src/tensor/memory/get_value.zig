const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;

const helpers = @import("../helpers.zig");

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;

pub fn getValue(
    comptime T: type,
    tensor: *Tensor(T),
    pipeline: *Pipeline,
    coor: []const u64,
    real_scalar: ?*T,
    imag_scalar: ?*T,
) !void {
    const is_complex = tensor.flags.is_complex;
    if (coor.len != tensor.dimensions.shape.len) {
        return tensor_module.Errors.InvalidCoordinates;
    } else if (real_scalar == null and imag_scalar == null) {
        return tensor_module.Errors.InvalidValue;
    } else if (imag_scalar != null and !is_complex) {
        return tensor_module.Errors.TensorDoesNotSupportComplexNumbers;
    }

    const buf_size: usize = @sizeOf(T) * (1 + @as(usize, @intFromBool(is_complex))); 

    var offset: usize = 0;
    for (tensor.dimensions.pitches, tensor.dimensions.shape, coor) |p, ds, c| {
        if (c >= ds) return tensor_module.Errors.InvalidCoordinates;

        offset += c * p;
    }
    offset *= @sizeOf(T);

    const prev_events = tensor.events_manager.getPrevEvents(.read);

    var new_event: cl.event.cl_event = undefined;
    var buf: [2]T = undefined;
    {
        try cl.buffer.read(
            command_queue.cmd,
            tensor.buffer,
            false,
            offset,
            buf_size,
            &buf,
            prev_events,
            &new_event,
        );
        errdefer |err| helpers.releaseEvent(new_event, err);

        _ = try tensor.events_manager.appendNewEvent(.read, prev_events, new_event, null);
    }

    try tensor.wait();

    if (real_scalar) |scalar| {
        scalar.* = buf[0];
    }

    if (imag_scalar) |scalar| {
        scalar.* = buf[1];
    }
}
