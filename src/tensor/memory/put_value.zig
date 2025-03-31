const cl = @import("opencl");

const core = @import("../../core/main.zig");

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const helpers = @import("../helpers.zig");

pub fn putValue(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const core.CommandQueue,
    coor: []const u64,
    real_scalar: ?T,
    imag_scalar: ?T,
) !void {
    const is_complex = tensor.flags.is_complex;

    if (coor.len != tensor.dimensions.shape.len) {
        return w_tensor.Errors.InvalidCoordinates;
    } else if (real_scalar == null and imag_scalar == null) {
        return w_tensor.Errors.InvalidValue;
    } else if (imag_scalar != null and !is_complex) {
        return w_tensor.Errors.TensorDoesNotSupportComplexNumbers;
    }

    const buf_size: usize = @sizeOf(T) * (1 + @as(usize, @intFromBool(is_complex))); 
    var buf: [2]T = undefined;

    buf[0] = real_scalar orelse 0;
    buf[1] = imag_scalar orelse 0;

    var offset: usize = 0;
    for (tensor.dimensions.pitches, tensor.dimensions.shape, coor) |p, ds, c| {
        if (c >= ds) return w_tensor.Errors.InvalidCoordinates;

        offset += c * p;
    }
    offset *= @sizeOf(T);

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    var new_event: cl.event.cl_event = undefined;
    {
        try cl.buffer.write(
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

        _ = try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
    }

    try tensor.wait();
}
