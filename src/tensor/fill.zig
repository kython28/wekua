const std = @import("std");
const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

pub fn constant(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
    real_scalar: ?T,
    imag_scalar: ?T,
) !void {
    const is_complex = tensor.is_complex;
    if (imag_scalar != null and !is_complex) {
        return w_tensor.Errors.InvalidValue;
    }

    var pattern: [2]T = undefined;

    pattern[0] = real_scalar orelse 0;
    pattern[1] = imag_scalar orelse 0;

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.fill(
        command_queue.cmd,
        tensor.buffer,
        &pattern,
        @sizeOf(T) * (1 + @as(usize, @intFromBool(is_complex))),
        0,
        tensor.size,
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

pub inline fn one(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
) !void {
    try constant(T, tensor, command_queue, @as(T, 1), null);
}

pub inline fn zeroes(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
) !void {
    try constant(T, tensor, command_queue, null, null);
}
