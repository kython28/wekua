const std = @import("std");
const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const utils = @import("../utils/utils.zig");

// const w_event = @import("utils/event.zig");
// const w_errors = @import("utils/errors.zig");

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
    if (imag_scalar != null and is_complex) |_| {
        return w_tensor.wTensorErrors.TensorDoesNotSupportComplexNumbers;
    }

    var pattern: [2]T = undefined;

    pattern[0] = real_scalar orelse 0;
    pattern[1] = imag_scalar orelse 0;

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.fill(
        command_queue.cmd,
        tensor.buffer,
        pattern.ptr,
        @sizeOf(T) * (1 + @as(usize, is_complex)),
        0,
        tensor.size,
        prev_events,
        &new_event,
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null, true);
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
