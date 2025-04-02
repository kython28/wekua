const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

inline fn get_random_number(comptime T: type, randprg: std.Random) T {
    if (T == f32 or T == f64) {
        return -1 + randprg.float(T) * 2;
    } else {
        return randprg.int(T);
    }
}

inline fn complex_mul(comptime T: type, re1: T, im1: T, re2: T, im2: T) struct { T, T } {
    const k1: T = re2 * (re1 + im1);
    const k2: T = re1 * (im2 - re2);
    const k3: T = im1 * (re2 + im2);
    return .{ k1 - k3, k1 + k2 };
}

inline fn get_index(
    op: wekua.blas.gemm.Operation,
    row: usize,
    col: usize,
    row_size: usize,
    col_size: usize,
) usize {
    return switch (op) {
        .no_transpose => row * col_size + col,
        .transpose => col * row_size + row,
    };
}

fn do_gemm(
    comptime T: type,
    comptime is_complex: bool,
    a: []T,
    b: []T,
    c: []T,
    m_size: usize,
    k_size: usize,
    n_size: usize,
    op_a: wekua.blas.gemm.Operation,
    op_b: wekua.blas.gemm.Operation,
    alpha: ?T,
    ialpha: ?T,
    beta: ?T,
    ibeta: ?T,
) !void {
    var alpha_scalar: T = undefined;
    var ialpha_scalar: T = undefined;

    if (ialpha) |s| {
        alpha_scalar = alpha orelse 0;
        ialpha_scalar = s;
    } else {
        alpha_scalar = alpha orelse 1;
        ialpha_scalar = ialpha orelse 0;
    }

    const beta_scalar = beta orelse 0;
    const ibeta_scalar = ibeta orelse 0;

    for (0..m_size) |row| {
        for (0..n_size) |col| {
            var c_index = row * n_size + col;
            if (is_complex) {
                c_index *= 2;

                const res = complex_mul(T, beta_scalar, ibeta_scalar, c[c_index], c[c_index + 1]);

                c[c_index] = res[0];
                c[c_index + 1] = res[1];
            } else {
                c[c_index] *= beta_scalar;
            }

            for (0..k_size) |i| {
                var a_index = get_index(op_a, row, i, m_size, k_size);
                var b_index = get_index(op_b, i, col, k_size, n_size);

                if (is_complex) {
                    a_index *= 2;
                    b_index *= 2;

                    const res = complex_mul(T, a[a_index], a[a_index + 1], b[b_index], b[b_index + 1]);
                    const res2 = complex_mul(T, alpha_scalar, ialpha_scalar, res[0], res[1]);

                    c[c_index] += res2[0];
                    c[c_index + 1] += res2[1];
                } else {
                    c[c_index] += alpha_scalar * a[a_index] * b[b_index];
                }
            }
        }
    }
}

fn test_gemm(
    comptime T: type,
    allocator: std.mem.Allocator,
    ctx: *wekua.core.Context,
    comptime is_complex: bool,
    vectors_enabled: bool,
    vectors_enabled2: bool,
    op_a: wekua.blas.gemm.Operation,
    op_b: wekua.blas.gemm.Operation,
    alpha: ?T,
    alphai: ?T,
    beta: ?T,
    betai: ?T,
    m_size: u64,
    k_size: u64,
    n_size: u64,
) !void {
    const w_cmd = &ctx.command_queues[0];
    if (!w_cmd.typeIsSupported(T)) {
        return;
    }

    const tensor = try wekua.Tensor(T).alloc(
        ctx,
        switch (op_a) {
            .no_transpose => &.{ m_size, k_size },
            .transpose => &.{ k_size, m_size },
        },
        .{
            .is_complex = is_complex,
            .vectors_enabled = vectors_enabled,
        },
    );
    defer tensor.release();

    const tensor2 = try wekua.Tensor(T).alloc(
        ctx,
        switch (op_b) {
            .no_transpose => &.{ k_size, n_size },
            .transpose => &.{ n_size, k_size },
        },
        .{
            .is_complex = is_complex,
            .vectors_enabled = vectors_enabled2,
        },
    );
    defer tensor2.release();

    const tensor3 = try wekua.Tensor(T).alloc(
        ctx,
        &.{ m_size, n_size },
        .{
            .is_complex = is_complex,
        },
    );
    defer tensor3.release();

    try wekua.tensor.random.fill(T, w_cmd, tensor, 0);
    try wekua.tensor.random.fill(T, w_cmd, tensor2, 0);

    if (beta != null or betai != null) {
        try wekua.tensor.random.fill(T, w_cmd, tensor3, 0);
    }

    const numbers1: []T = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers2: []T = try allocator.alloc(T, tensor2.dimensions.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    const expected_result: []T = try allocator.alloc(T, tensor3.dimensions.number_of_elements_without_padding);
    defer allocator.free(expected_result);

    const numbers3: []T = try allocator.alloc(T, tensor3.dimensions.number_of_elements_without_padding);
    defer allocator.free(numbers3);

    try wekua.tensor.memory.writeToBuffer(T, tensor, w_cmd, numbers1);
    try wekua.tensor.memory.writeToBuffer(T, tensor2, w_cmd, numbers2);
    try wekua.tensor.memory.writeToBuffer(T, tensor3, w_cmd, expected_result);

    try wekua.blas.gemm.perform(
        T,
        w_cmd,
        alpha,
        alphai,
        tensor,
        op_a,
        tensor2,
        op_b,
        beta,
        betai,
        tensor3,
    );

    try do_gemm(
        T,
        is_complex,
        numbers1,
        numbers2,
        expected_result,
        m_size,
        k_size,
        n_size,
        op_a,
        op_b,
        alpha,
        alphai,
        beta,
        betai,
    );

    try wekua.tensor.memory.writeToBuffer(T, tensor3, w_cmd, numbers3);
    const eps: T = @floatCast(comptime std.math.floatEps(f32) * 100);
    for (expected_result, numbers3) |expected, result| {
        try std.testing.expectApproxEqAbs(expected, result, eps);
    }
}

fn test_gemm_kernel(
    min_random: u64,
    max_random: u64,
    padding: u64,
    comptime test_complex: bool,
    comptime types: []const type
) !void {
    const allocator = std.testing.allocator;
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    const bool_array = [_]bool{ false, true };
    const op_array = [_]wekua.blas.gemm.Operation{
        wekua.blas.gemm.Operation.no_transpose,
        wekua.blas.gemm.Operation.transpose,
    };

    // Hard to test with integers due overflow :-/
    inline for (types) |T| {
        // TODO: This look very ugly, find a better way to do this
        inline for (bool_array) |is_complex| {
            if (is_complex and !test_complex) continue;
            for (bool_array) |vectors_enabled| {
                for (bool_array) |vectors_enabled2| {
                    for (op_array) |op_a| {
                        for (op_array) |op_b| {
                            var m_size = randprg.intRangeAtMost(u64, min_random, max_random);
                            var k_size = randprg.intRangeAtMost(u64, min_random, max_random);
                            var n_size = randprg.intRangeAtMost(u64, min_random, max_random);

                            if (m_size % padding != 0) m_size += padding - (m_size % padding);
                            if (k_size % padding != 0) k_size += padding - (k_size % padding);
                            if (n_size % padding != 0) n_size += padding - (n_size % padding);

                            test_gemm(
                                T,
                                allocator,
                                ctx,
                                is_complex,
                                vectors_enabled,
                                vectors_enabled2,
                                op_a,
                                op_b,
                                null,
                                null,
                                null,
                                null,

                                m_size,
                                k_size,
                                n_size,
                            ) catch |err| {
                                std.log.warn(
                                    \\ An error while testing gemm with is_complex: {}, vectors_enabled: {},
                                    \\ vectors_enabled2: {}, op_a: {}, op_b: {}: {s}
                                , .{
                                    is_complex,
                                    vectors_enabled,
                                    vectors_enabled2,
                                    op_a,
                                    op_b,
                                    @errorName(err),
                                });
                                return err;
                            };

                            const alpha = get_random_number(T, randprg);
                            const ialpha = get_random_number(T, randprg);

                            const beta = get_random_number(T, randprg);
                            const ibeta = get_random_number(T, randprg);

                            test_gemm(
                                T,
                                allocator,
                                ctx,
                                is_complex,
                                vectors_enabled,
                                vectors_enabled2,
                                op_a,
                                op_b,
                                alpha,
                                ialpha,
                                beta,
                                ibeta,

                                m_size,
                                k_size,
                                n_size,
                            ) catch |err| {
                                std.log.warn(
                                    \\ An error while testing gemm with is_complex: {}, vectors_enabled: {},
                                    \\ vectors_enabled2: {}, op_a: {}, op_b: {}: {s} -
                                    \\ alpha: {d}, ialpha: {d}, beta: {d}, ibeta: {d}
                                , .{
                                    is_complex,
                                    vectors_enabled,
                                    vectors_enabled2,
                                    op_a,
                                    op_b,
                                    @errorName(err),
                                    alpha,
                                    ialpha,
                                    beta,
                                    ibeta,
                                });
                                return err;
                            };
                        }
                    }
                }
            }
        }
    }
}

test {
    try test_gemm_kernel(5, 20, 1, true, &.{ f32, f64 });
    try test_gemm_kernel(32, 60, 4, false, &.{ f64 });
    try test_gemm_kernel(64, 120, 8, false, &.{ f64 });
    try test_gemm_kernel(128, 240, 16, false, &.{ f64 });
    try test_gemm_kernel(256, 300, 32, false, &.{ f64 });
}
