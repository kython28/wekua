const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_event = @import("../../utils/event.zig");
const w_errors = @import("../../utils/errors.zig");

const dtypes = @import("../../utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wScalar = dtypes.wScalar;
const wTensorDtype = dtypes.wTensorDtype;

const put_value_resources = struct {
    pattern: []u8
};

fn put_value_callback(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    const resources: *put_value_resources = @alignCast(@ptrCast(user_data.?));
    allocator.free(resources.pattern);
    allocator.destroy(resources);
}

pub fn put_value(
    command_queue: wCommandQueue, tensor: wTensor, coor: []const u64,
    real_scalar: ?wScalar, imag_scalar: ?wScalar
) !void {
    const is_complex = tensor.is_complex;
    if (coor.len != tensor.shape.len) {
        return w_errors.errors.InvalidCoordinates;
    }
    else if (real_scalar == null and imag_scalar == null) {
        return w_errors.errors.InvalidValue;
    }
    else if (imag_scalar != null and !is_complex) {
        return w_errors.errors.TensorIsnotComplex;
    }

    const allocator = command_queue.allocator;
    const dtype = tensor.dtype;

    const dtype_size = dtypes.get_dtype_size(dtype);
    var pattern_size: usize = dtype_size;
    if (is_complex) {
        pattern_size += dtype_size;
    }

    const pattern: []u8 = try allocator.alloc(u8, pattern_size);
    errdefer allocator.free(pattern);

    if (real_scalar) |scalar| {
        if (dtype != @as(wTensorDtype, scalar)) {
            return w_errors.errors.InvalidScalarDtype;
        }

        @memcpy(
            pattern[0..dtype_size],
            @as([*]const u8, @ptrCast(&scalar))[0..dtype_size]
        );
    }else{
        @memset(pattern[0..dtype_size], 0);
    }

    if (imag_scalar) |scalar| {
        if (dtype != @as(wTensorDtype, scalar)) {
            return w_errors.errors.InvalidScalarDtype;
        }

        @memcpy(
            pattern[dtype_size..pattern_size],
            @as([*]const u8, @ptrCast(&scalar))[0..dtype_size]
        );
    }else{
        @memset(pattern[dtype_size..], 0);
    }

    const shape = tensor.shape;

    var stride: usize = @intCast(tensor.number_of_elements);
    var offset: usize = 0;

    const last_index = shape.len - 1;
    for (shape[0..last_index], coor[0..last_index]) |v, c| {
        stride /= v;
        offset += c * stride;
    }
    offset += coor[last_index] * (1 + @as(usize, @intCast(@intFromBool(is_complex))));
    offset *= dtype_size;

    const prev_events = w_event.acquire_tensor(tensor, .write);
    defer tensor.mutex.unlock();

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.write(
        command_queue.cmd, tensor.buffer, false, offset, pattern_size, pattern.ptr, prev_events,
        &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    const resources: *put_value_resources = try allocator.create(put_value_resources);
    errdefer allocator.destroy(resources);
    resources.pattern = pattern;

    try w_event.register_new_event_to_single_tensor(command_queue, tensor, &put_value_callback, resources, new_event, .write);
}
