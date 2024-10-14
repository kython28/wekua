const std = @import("std");
const cl = @import("opencl");

const work_items = @import("../utils/work_items.zig");

const w_command_queue = @import("../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_event = @import("../tensor/utils/event.zig");
const w_errors = @import("../tensor/utils/errors.zig").errors;
const w_kernel = @import("../core/kernel.zig");

const dtypes = @import("../tensor/utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wTensorDtype = dtypes.wTensorDtype;
const wScalar = dtypes.wScalar;

const validations = @import("../tensor/utils/validations.zig");

const gemm_cl_kernel: []const u8 = @embedFile("kernels/gemm.cl");

const axpy_resources = struct {
    prev_events: []cl.event.cl_event
};

fn release_events_array(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    if (user_data) |data| {
        const resources: *axpy_resources = @ptrCast(@alignCast(data));
        allocator.free(resources.prev_events);
        allocator.destroy(resources);
    }
}

pub const Operation = enum {
    Transpose, NoTranspose,
    ConjugateTranspose // TODO
};

inline fn get_scalar(scalar: ?wScalar, dtype: wTensorDtype) !wScalar {
    if (scalar) |v| {
        if (dtype != @as(wTensorDtype, v)) {
            return w_errors.InvalidScalarDtype;
        }

        return v;
    }

    return dtypes.initialize_scalar(dtype, 0);
}

fn validate_tensors(A: wTensor, B: wTensor, C: wTensor, opA: Operation, opB: Operation) !void {
    if (opA == .ConjugateTranspose or opB == .ConjugateTranspose) {
        @panic("ConjugateTranspose is not implemented yet");
    }

    const a_shape = A.shape;
    const b_shape = B.shape;
    const c_shape = C.shape;

    if (c_shape.len < 2) return w_errors.InvalidValue;
    try validations.eql_tensors_dimensions(A, B);
    try validations.eql_tensors_dimensions(B, C);

    const ndim = a_shape.len;
    const a_m = a_shape[ndim - 2];
    const a_k = a_shape[ndim - 1];

    const b_k = b_shape[ndim - 2];
    const b_n = b_shape[ndim - 1];

    const c_m = c_shape[ndim - 2];
    const c_n = c_shape[ndim - 1];

    switch (opA) {
        .Transpose => {
            if (opB == .NoTranspose and a_m == b_k and b_n == c_n) {
                return w_errors.InvalidValue;
            }else if (a_m == b_n and b_k == c_n) {
                return w_errors.InvalidValue;
            }

            if (a_k != c_m) return w_errors.InvalidValue;
        },
        .NoTranspose => {
            if (opB == .NoTranspose and a_k == b_k and b_n == c_n) {
                return w_errors.InvalidValue;
            }else if (a_k == b_n and b_k == c_n) {
                return w_errors.InvalidValue;
            }

            if (a_m != c_m) return w_errors.InvalidValue;
        },
        .ConjugateTranspose => unreachable
    }
}

pub fn gemm(
    command_queue: wCommandQueue, alpha: ?wScalar, alphai: ?wScalar, A: wTensor, opA: Operation,
    B: wTensor, opB: Operation, beta: ?wScalar, betai: ?wScalar, C: wTensor
) !void {
    if ((alpha == null and alphai == null) or (beta == null and betai == null)) return w_errors.InvalidValue;
    try validate_tensors(A, B, C, opA, opB);

    const dtype = A.dtype;
    const real_alpha_scalar: wScalar = try get_scalar(alpha, dtype);
    const imag_alpha_scalar: wScalar = try get_scalar(alpha, dtype);

    const dims = C.vl_shape;
    const ndim = dims.len;
    var global_items: [3]u64 = .{
        dims[ndim - 2], dims[ndim - 1] / (1 + @as(u64, @intFromBool(C.is_complex))), 1
    };

    if (ndim > 2) {
        for (dims[0..(ndim - 2)]) |v| global_items[2] *= v;
    }

    const use_optimized_gemm: bool = (
        (@intFromBool(A.vectors_enabled)&@intFromBool(B.vectors_enabled)&@intFromBool(C.vectors_enabled)) == 1 and
        (global_items[0]%2) == 0 and (global_items[1]%2) == 0
    );

    if (use_optimized_gemm) {
        inline for (global_items[0..2]) |*v| v.* /= 2;
    }

    var local_work_items: [3]u64 = undefined;
    try work_items.get(
        &global_items, &local_work_items, command_queue.max_work_group_size
    );

    const allocator = command_queue.allocator;
    const kernel = try w_kernel.get_cl_kernel(
        command_queue, A, .GEMM, "gemm", gemm_cl_kernel, null
    );

    const cmd = command_queue.cmd;

    const a_prev_events = w_event.acquire_tensor(A, .read);
    defer A.mutex.unlock();

    const b_prev_events = w_event.acquire_tensor(B, .read);
    defer B.mutex.unlock();

    const c_prev_events = w_event.acquire_tensor(C, .write);
    defer C.mutex.unlock();

    const prev_events = try w_event.concatenate_events(allocator, &.{
        a_prev_events, b_prev_events, c_prev_events
    });
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);
    const scalar_size = dtypes.get_dtype_size(dtype);

    var global: u64 = C.number_of_vectors;
    var work_items: u64 = A.work_item_for_all_vectors[command_queue.wekua_id];

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, scalar_size, &real_scalar);
    if (x.is_complex) {
        const imag_scalar: wScalar = try get_scalar(beta, dtype);
        global /= 2;
        work_items /= 2;
        try set_arg(kernel, 3, scalar_size, &imag_scalar);
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd, kernel, null, @as([*]const u64, @ptrCast(&global))[0..1],
        @as([*]const u64, @ptrCast(&work_items))[0..1],
        null, &new_event
    );
    errdefer {
        cl.event.wait(new_event) catch unreachable;
        cl.event.release(new_event) catch unreachable;
    }

    var resources: ?*axpy_resources = null;
    if (prev_events) |v| {
        resources = try allocator.create(axpy_resources);
        resources.?.prev_events = v;
    }
    errdefer {
        if (resources) |v| allocator.destroy(v);
    }

    try w_event.register_new_event_to_multiple_tensors(
        command_queue, &.{x, y}, &release_events_array, resources, new_event, &.{.read, .write}
    );
}
