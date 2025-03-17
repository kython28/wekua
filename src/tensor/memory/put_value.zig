const std = @import("std");
const cl = @import("opencl");

const core = @import("../../core/main.zig");

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

pub fn putValue(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const core.CommandQueue,
    coor: []const u64,
    real_scalar: ?T,
    imag_scalar: ?T,
) !void {
    const is_complex = tensor.is_complex;

    if (coor.len != tensor.shape.len) {
        return w_tensor.Errors.InvalidCoordinates;
    } else if (real_scalar == null and imag_scalar == null) {
        return w_tensor.Errors.InvalidValue;
    } else if (imag_scalar != null and !is_complex) {
        return w_tensor.Errors.TensorDoesNotSupportComplexNumbers;
    }

    var pattern: [2]T = undefined;

    pattern[0] = real_scalar orelse 0;
    pattern[1] = imag_scalar orelse 0;

    var stride: usize = @intCast(tensor.number_of_elements);
    var offset: usize = 0;

    const shape = tensor.shape;
    const last_index = shape.len - 1;
    for (shape[0..last_index], coor[0..last_index]) |v, c| {
        stride /= v;
        offset += c * stride;
    }
    offset += coor[last_index] * (1 + @as(usize, @intFromBool(is_complex)));
    offset *= @sizeOf(T);

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    var new_event: cl.event.cl_event = undefined;
    {
        try cl.buffer.write(
            command_queue.cmd,
            tensor.buffer,
            false,
            offset,
            @sizeOf(T) * (1 + @as(usize, @intFromBool(is_complex))),
            &pattern,
            prev_events,
            &new_event,
        );
        errdefer |err| {
            cl.event.wait(new_event) catch |err2| {
                std.debug.panic(
                    "An error ocurred ({s}) while waiting for new event and dealing with another error ({s})", .{
                        @errorName(err2), @errorName(err)
                    }
                );
            };
            cl.event.release(new_event);
        }

        try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
    }

    try tensor.wait();
}
