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

const utils = @import("utils.zig");

fn write_value_to_scalar(pattern: []u8, scalar: *wScalar, dtype: wTensorDtype) void {
    const tensor_dtype_fields = @typeInfo(wTensorDtype).Enum.fields;
    scalar.* = blk: inline for (tensor_dtype_fields) |field| {
        if (@field(wTensorDtype, field.name) == dtype) {
            var new_scalar: wScalar = dtypes.initialize_scalar(dtype, undefined);
            @memcpy(
                @as([*]u8, @ptrCast(&@field(new_scalar, field.name)))[0..pattern.len],
                pattern
            );
            break :blk new_scalar;
        }
    }else{ unreachable; };
}

pub fn get_value(
    command_queue: wCommandQueue, tensor: wTensor, coor: []const u64,
    real_scalar: ?*wScalar, imag_scalar: ?*wScalar
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
    defer allocator.free(pattern);

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

    const prev_events = w_event.acquire_tensor(tensor, .read);
    const tensor_mutex = &tensor.mutex;
    errdefer tensor_mutex.unlock();

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.read(
        command_queue.cmd, tensor.buffer, false, offset, pattern_size,
        pattern.ptr, prev_events, &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    var cond = std.Thread.Condition{};
    try w_event.register_new_event(
        command_queue, tensor, &utils.signal_condition_callback, &cond, new_event, .read
    );
    cond.wait(tensor_mutex);
    tensor_mutex.unlock();

    if (real_scalar) |scalar| {
        write_value_to_scalar(pattern[0..dtype_size], scalar, dtype);
    }

    if (imag_scalar) |scalar| {
        write_value_to_scalar(pattern[dtype_size..pattern_size], scalar, dtype);
    }
}
