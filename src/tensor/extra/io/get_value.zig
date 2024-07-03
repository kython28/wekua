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

fn get_value_callback(_: *const std.mem.Allocator, user_data: ?*anyopaque) void {
    const cond: *std.Thread.Condition = @alignCast(@ptrCast(user_data.?));
    cond.signal();
}

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
    if (coor.len != tensor.shape.len) {
        return w_errors.errors.InvalidCoordinates;
    }

    const allocator = command_queue.allocator;
    const dtype = tensor.dtype;

    const dtype_size = dtypes.get_dtype_size(dtype);
    var pattern_size: usize = 0;
    if (real_scalar) |_| {
        pattern_size += dtype_size;
    }

    if (imag_scalar) |_| {
        if (!tensor.is_complex) {
            return w_errors.errors.TensorIsnotComplex;
        }
        pattern_size += dtype_size;
    }

    if (pattern_size == 0) {
        return w_errors.errors.InvalidValue;
    }

    const pattern: []u8 = try allocator.alloc(u8, pattern_size);
    defer allocator.free(pattern);

    const shape = tensor.shape;

    var stride: usize = @intCast(tensor.number_of_elements);
    var offset: usize = 0;

    for (shape[0..(shape.len - 1)], coor[0..(shape.len - 1)]) |v, c| {
        stride /= v;
        offset += c * stride;
    }
    offset += coor[shape.len - 1]; 

    
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
    try w_event.register_new_event(command_queue, tensor, &get_value_callback, &cond, new_event, .read);
    cond.wait(tensor_mutex);
    tensor_mutex.unlock();

    if (real_scalar) |scalar| {
        write_value_to_scalar(pattern[0..dtype_size], scalar, dtype);
    }

    if (imag_scalar) |scalar| {
        write_value_to_scalar(pattern[dtype_size..pattern_size], scalar, dtype);
    }
}
