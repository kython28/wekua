const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_empty = @import("../empty.zig");
const w_event = @import("../utils/event.zig");
const w_errors = @import("../utils/errors.zig");

const dtypes = @import("../utils/dtypes.zig");
const wTensor = dtypes.wTensor;

const fill_resources = struct {
    pattern: []u8
};

fn fill_callback(allocator: *const std.mem.Allocator, user_data: ?*anyopaque) void {
    const resources: *fill_resources = @alignCast(@ptrCast(user_data.?));
    allocator.free(resources.pattern);
    allocator.destroy(resources);
}

pub fn fill(
    command_queue: wCommandQueue, tensor: wTensor,
    real_scalar: ?*const anyopaque, imag_scalar: ?*const anyopaque
) !void {
    const allocator = command_queue.allocator;
    const dtype = tensor.dtype;

    const dtype_size = dtypes.get_dtype_size(dtype);
    var pattern_size: usize = dtype_size;
    if (imag_scalar) |_| {
        if (!tensor.is_complex) {
            return w_errors.errros.TensorIsnotComplex;
        }
        pattern_size += dtype_size;
    }

    const pattern: []u8 = try allocator.alloc(u8, pattern_size);
    errdefer allocator.free(pattern);

    if (real_scalar) |scalar| {
        @memcpy(
            pattern,
            @as([*]const u8, @ptrCast(scalar))[0..dtype_size]
        );
    }else{
        @memset(pattern[0..dtype_size], 0);
    }

    if (imag_scalar) |scalar| {
        @memcpy(
            pattern[dtype_size..pattern_size],
            @as([*]const u8, @ptrCast(scalar))[0..dtype_size]
        );
    }else{
        @memset(pattern[dtype_size..], 0);
    }

    const prev_event = w_event.acquire_tensor(tensor, .write);
    defer tensor.mutex.unlock();

    var events_to_wait: ?[]const cl.event.cl_event = null;
    if (prev_event != null) {
        events_to_wait = @as([*]const cl.event.cl_event, @ptrCast(&prev_event.?))[0..1];
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.fill(
        command_queue.cmd, tensor.buffer, pattern.ptr, pattern_size,
        0, tensor.size, events_to_wait,
        &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    const resources: *fill_resources = try allocator.create(fill_resources);
    errdefer allocator.destroy(resources);
    resources.pattern = pattern;

    try w_event.register_new_event(command_queue, tensor, &fill_callback, resources, new_event, .write);
}
