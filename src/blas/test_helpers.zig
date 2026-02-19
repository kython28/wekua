const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const gemm_module = @import("gemm.zig");
const gemm = gemm_module.gemm;
const Operation = gemm_module.Operation;
const PackedTensors = gemm_module.PackedTensors;

const memory = tensor_module.memory;
const fill = tensor_module.fill;
const identity_fn = tensor_module.identity;

const testing = std.testing;

pub fn castInt(comptime T: type, val: anytype) T {
    return switch (@typeInfo(T)) {
        .float => @floatFromInt(val),
        .int => @intCast(val),
        else => unreachable,
    };
}

pub fn castComplex(comptime T: type, real: anytype, imag: anytype) T {
    const Scalar = core.types.getType(T);
    return .{
        .real = castInt(Scalar, real),
        .imag = castInt(Scalar, imag),
    };
}

fn makeDataValue(comptime T: type, index: usize) T {
    if (comptime core.types.isComplex(T)) {
        return castComplex(T, index + 1, 0);
    } else {
        return castInt(T, index + 1);
    }
}

fn computeExpected(comptime T: type, data_val: T, alpha: ?T, beta: ?T) T {
    const is_complex = comptime core.types.isComplex(T);
    var expected: T = undefined;

    if (alpha) |a| {
        if (is_complex) {
            expected = .{
                .real = a.real * data_val.real - a.imag * data_val.imag,
                .imag = a.real * data_val.imag + a.imag * data_val.real,
            };
        } else {
            expected = a * data_val;
        }
    } else {
        expected = data_val;
    }

    if (beta) |b| {
        if (is_complex) {
            // C was filled with ones = {1, 0}; beta * {1, 0} = {b.real, b.imag}
            expected.real += b.real;
            expected.imag += b.imag;
        } else {
            expected += b; // beta * 1
        }
    }

    return expected;
}

fn expectEqualValue(comptime T: type, expected: T, actual: T) !void {
    if (comptime core.types.isComplex(T)) {
        try testing.expectEqual(expected.real, actual.real);
        try testing.expectEqual(expected.imag, actual.imag);
    } else {
        try testing.expectEqual(expected, actual);
    }
}

/// Tests C = alpha * op(A) * op(I) + beta * C_old
/// A shape: if op_a=no_transpose → (m, k), if op_a=transpose → (k, m)
/// Identity shape: (k, k), C shape: (m, k)
pub fn testGemmATimesIdentity(
    comptime T: type,
    context: *const core.Context,
    pipeline: *Pipeline,
    m: u64,
    k: u64,
    op_a: Operation,
    op_b: Operation,
    use_packing: bool,
    alpha: ?T,
    beta: ?T,
) !void {
    const allocator = testing.allocator;
    const config = tensor_module.CreateConfig{};

    const a_shape: [2]u64 = if (op_a == .transpose) .{ k, m } else .{ m, k };
    const ident_shape = [_]u64{ k, k };
    const c_shape = [_]u64{ m, k };

    const a = try Tensor(T).alloc(context, pipeline, &a_shape, config);
    defer a.release(pipeline);

    const ident = try Tensor(T).alloc(context, pipeline, &ident_shape, config);
    defer ident.release(pipeline);

    const c_mat = try Tensor(T).alloc(context, pipeline, &c_shape, config);
    defer c_mat.release(pipeline);

    const a_elems: usize = @intCast(a_shape[0] * a_shape[1]);
    const a_buf = try allocator.alloc(T, a_elems);
    defer allocator.free(a_buf);

    for (a_buf, 0..) |*val, i| {
        val.* = makeDataValue(T, i);
    }

    try memory.readFromBuffer(T, pipeline, a, a_buf);
    try identity_fn(T, pipeline, ident);

    if (beta != null) {
        try fill.one(T, pipeline, c_mat);
    }

    var packed_tensors: ?*PackedTensors(T) = null;
    if (use_packing) {
        packed_tensors = try PackedTensors(T).init(pipeline, c_mat, k, true);
    }
    defer if (packed_tensors) |pt| pt.deinit(pipeline);

    try gemm(T, pipeline, alpha, a, op_a, ident, op_b, beta, c_mat, packed_tensors);

    const c_elems: usize = @intCast(m * k);
    const result = try allocator.alloc(T, c_elems);
    defer allocator.free(result);

    try memory.writeToBuffer(T, pipeline, c_mat, result);
    pipeline.waitAndCleanup();

    const m_usize: usize = @intCast(m);
    const k_usize: usize = @intCast(k);

    for (0..m_usize) |i| {
        for (0..k_usize) |j| {
            const idx = i * k_usize + j;
            // op(A) * op(I) = op(A) since I^T = I
            // no_transpose: result[i][j] = A[i][j] = a_buf[i*k + j]
            // transpose: A stored as (k,m), result[i][j] = A^T[i][j] = A[j][i] = a_buf[j*m + i]
            const data_idx: usize = if (op_a == .transpose)
                j * m_usize + i
            else
                idx;

            const data_val = makeDataValue(T, data_idx);
            const expected = computeExpected(T, data_val, alpha, beta);
            try expectEqualValue(T, expected, result[idx]);
        }
    }
}

/// Tests C = alpha * op(I) * op(B) + beta * C_old
/// Identity shape: (m, m), B shape: if op_b=no_transpose → (m, n), if op_b=transpose → (n, m)
/// C shape: (m, n)
pub fn testGemmIdentityTimesB(
    comptime T: type,
    context: *const core.Context,
    pipeline: *Pipeline,
    m: u64,
    n: u64,
    op_a: Operation,
    op_b: Operation,
    use_packing: bool,
    alpha: ?T,
    beta: ?T,
) !void {
    const allocator = testing.allocator;
    const config = tensor_module.CreateConfig{};

    const ident_shape = [_]u64{ m, m };
    const b_shape: [2]u64 = if (op_b == .transpose) .{ n, m } else .{ m, n };
    const c_shape = [_]u64{ m, n };

    const ident = try Tensor(T).alloc(context, pipeline, &ident_shape, config);
    defer ident.release(pipeline);

    const b = try Tensor(T).alloc(context, pipeline, &b_shape, config);
    defer b.release(pipeline);

    const c_mat = try Tensor(T).alloc(context, pipeline, &c_shape, config);
    defer c_mat.release(pipeline);

    const b_elems: usize = @intCast(b_shape[0] * b_shape[1]);
    const b_buf = try allocator.alloc(T, b_elems);
    defer allocator.free(b_buf);

    for (b_buf, 0..) |*val, i| {
        val.* = makeDataValue(T, i);
    }

    try memory.readFromBuffer(T, pipeline, b, b_buf);
    try identity_fn(T, pipeline, ident);

    if (beta != null) {
        try fill.one(T, pipeline, c_mat);
    }

    var packed_tensors: ?*PackedTensors(T) = null;
    if (use_packing) {
        packed_tensors = try PackedTensors(T).init(pipeline, c_mat, m, true);
    }
    defer if (packed_tensors) |pt| pt.deinit(pipeline);

    try gemm(T, pipeline, alpha, ident, op_a, b, op_b, beta, c_mat, packed_tensors);

    const c_elems: usize = @intCast(m * n);
    const result = try allocator.alloc(T, c_elems);
    defer allocator.free(result);

    try memory.writeToBuffer(T, pipeline, c_mat, result);
    pipeline.waitAndCleanup();

    const m_usize: usize = @intCast(m);
    const n_usize: usize = @intCast(n);

    for (0..m_usize) |i| {
        for (0..n_usize) |j| {
            const idx = i * n_usize + j;
            // op(I) * op(B) = op(B) since I^T = I
            // no_transpose: result[i][j] = B[i][j] = b_buf[i*n + j]
            // transpose: B stored as (n,m), result[i][j] = B^T[i][j] = B[j][i] = b_buf[j*m + i]
            const data_idx: usize = if (op_b == .transpose)
                j * m_usize + i
            else
                idx;

            const data_val = makeDataValue(T, data_idx);
            const expected = computeExpected(T, data_val, alpha, beta);
            try expectEqualValue(T, expected, result[idx]);
        }
    }
}
