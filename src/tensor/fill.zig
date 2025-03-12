const std = @import("std");
const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const utils = @import("../utils/utils.zig");

const w_event = @import("utils/event.zig");
const w_errors = @import("utils/errors.zig");

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

pub fn constant(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
    real_scalar: ?T,
    imag_scalar: ?T,
) !void {
    const allocator = command_queue.allocator;

    const is_complex = tensor.is_complex;
    if (imag_scalar != null and is_complex) |_| {
        return w_tensor.wTensorErrors.TensorDoesNotSupportComplexNumbers;
    }

    const Resources = packed struct {
        size: usize,
        pattern: [2]T
    };

    const resources = try allocator.create(Resources);
    errdefer allocator.free(resources);

    resources.size = @sizeOf(Resources);

    const pattern: []T = &resources.pattern;

    pattern[0] = real_scalar orelse 0;
    pattern[1] = imag_scalar orelse 0;

    const prev_events = w_event.acquire_tensor(tensor, .write);
    defer tensor.mutex.unlock();

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

    try w_event.register_new_event_to_single_tensor(
        command_queue,
        tensor,
        &utils.release_temporal_resource_callback,
        resources,
        new_event,
        .write,
    );
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
