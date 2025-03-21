const cl = @import("opencl");

const core = @import("../../core/main.zig");
const helpers = @import("../helpers.zig");

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

pub fn getValue(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const core.CommandQueue,
    coor: []const u64,
    real_scalar: ?*T,
    imag_scalar: ?*T,
) !void {
    const is_complex = tensor.is_complex;
    if (coor.len != tensor.shape.len) {
        return w_tensor.Errors.InvalidCoordinates;
    } else if (real_scalar == null and imag_scalar == null) {
        return w_tensor.Errors.InvalidValue;
    } else if (imag_scalar != null and !is_complex) {
        return w_tensor.Errors.TensorDoesNotSupportComplexNumbers;
    }

    var buf_size: usize = @sizeOf(T);
    if (is_complex) {
        buf_size += @sizeOf(T);
    }

    var stride: usize = @intCast(tensor.number_of_elements);
    var offset: usize = 0;

    const shape = tensor.shape;
    const last_index = shape.len - 1;
    for (shape[0..last_index], coor[0..last_index]) |v, c| {
        stride /= v;
        offset += c * stride;
    }
    offset += coor[last_index] * (1 + @as(usize, @intCast(@intFromBool(is_complex))));
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
        errdefer |err| helpers.release_event(new_event, err);

        try tensor.events_manager.appendNewEvent(.read, prev_events, new_event, null);
    }

    try tensor.wait();

    if (real_scalar) |scalar| {
        scalar.* = buf[0];
    }

    if (imag_scalar) |scalar| {
        scalar.* = buf[1];
    }
}
