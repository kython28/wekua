const std = @import("std");
const builtin = @import("builtin");

const wekua = @import("wekua");
const cl = wekua.opencl;
const utils = @import("utils.zig");

pub const name: []const u8 = "GEMM";
pub const starting_point: u64 = 4;

const niterations = switch (builtin.mode) {
    .Debug => 1,
    else => 10,
};

const PreferredType = f32;
// const PreferredType = f64;
const WekuaCPreferredType = utils.wekua_c.WEKUA_DTYPE_FLOAT;
// const WekuaCPreferredType = utils.wekua_c.WEKUA_DTYPE_DOUBLE;

fn run_openblas_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    const gemm_func = if (PreferredType == f32) utils.openblas.cblas_sgemm else utils.openblas.cblas_dgemm;

    const a = try allocator.dupe(PreferredType, buf1);
    defer allocator.free(a);

    const b = try allocator.dupe(PreferredType, buf2);
    defer allocator.free(b);

    const c = try allocator.dupe(PreferredType, buf3);
    defer allocator.free(c);

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (alphas, betas) |alpha, beta| {
        gemm_func(
            utils.openblas.CblasRowMajor,
            utils.openblas.CblasNoTrans,
            utils.openblas.CblasTrans,
            @intCast(size),
            @intCast(size),
            @intCast(size),
            alpha,
            a.ptr,
            @intCast(size),
            b.ptr,
            @intCast(size),
            beta,
            c.ptr,
            @intCast(size),
        );
    }
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(@divTrunc(end_ts - start_ts, niterations))) / 1000.0;

    return total_diff;
}

fn run_old_wekua_test(
    device_type: cl.device.enums.device_type,
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    const ctx: utils.wekua_c.wekuaContext = utils.wekua_c.createSomeWekuaContext(
        @intFromEnum(device_type),
        1,
        0,
    ) orelse return error.OutOfMemory;
    defer utils.wekua_c.freeWekuaContext(ctx);

    const a = utils.wekua_c.wekuaAllocMatrix(ctx, size, size, WekuaCPreferredType) orelse return error.OutOfMemory;
    defer utils.wekua_c.wekuaFreeMatrix(a, 0, null);

    const b = utils.wekua_c.wekuaAllocMatrix(ctx, size, size, WekuaCPreferredType) orelse return error.OutOfMemory;
    defer utils.wekua_c.wekuaFreeMatrix(b, 0, null);

    const c = utils.wekua_c.wekuaAllocMatrix(ctx, size, size, WekuaCPreferredType) orelse return error.OutOfMemory;
    defer utils.wekua_c.wekuaFreeMatrix(c, 0, null);

    var ret = utils.wekua_c.wekuaMatrixCopyBuffer(a, buf1.ptr, null);
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);

    ret = utils.wekua_c.wekuaMatrixCopyBuffer(b, buf2.ptr, null);
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);

    ret = utils.wekua_c.wekuaMatrixCopyBuffer(c, buf3.ptr, null);
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);

    _ = utils.wekua_c.compileKernel(ctx, utils.wekua_c.WEKUA_KERNEL_GEMM, WekuaCPreferredType, 0);

    const events: []cl.event.cl_event = try allocator.alloc(cl.event.cl_event, niterations);
    defer allocator.free(events);

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (alphas, betas) |alpha, beta| {
        ret = utils.wekua_c.wekuaBlasGemm(
            @constCast(&alpha),
            null,
            0,
            a,
            1,
            b,
            @constCast(&beta),
            null,
            c,
            0,
            null,
        );
        if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);
    }
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(@divTrunc(end_ts - start_ts, niterations))) / 1000.0;

    return total_diff;
}

inline fn run_old_wekua_cpu_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    return run_old_wekua_test(.cpu, allocator, size, alphas, betas, buf1, buf2, buf3);
}

inline fn run_old_wekua_gpu_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    return run_old_wekua_test(.gpu, allocator, size, alphas, betas, buf1, buf2, buf3);
}

fn run_wekua_test(
    device_type: cl.device.enums.device_type,
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    const ctx = try wekua.core.Context.create_from_best_device(allocator, null, device_type);
    defer ctx.release();

    const cmd = &ctx.command_queues[0];

    const FloatTensor = wekua.Tensor(PreferredType);

    const a = try FloatTensor.alloc(ctx, &.{ size, size }, .{});
    defer a.release();

    const b = try FloatTensor.alloc(ctx, &.{ size, size }, .{});
    defer b.release();

    const c = try FloatTensor.alloc(ctx, &.{ size, size }, .{});
    defer c.release();

    // At the first time, we need to compile the kernels
    try wekua.blas.gemm.perform(
        PreferredType,
        cmd,
        1.0,
        null,
        a,
        .no_transpose,
        b,
        .transpose,
        0.0,
        null,
        c,
    );

    try wekua.tensor.memory.readFromBuffer(PreferredType, a, cmd, buf1);
    try wekua.tensor.memory.readFromBuffer(PreferredType, b, cmd, buf2);
    try wekua.tensor.memory.readFromBuffer(PreferredType, c, cmd, buf3);

    try a.wait();
    try b.wait();
    try c.wait();

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (alphas, betas) |alpha, beta| {
        try wekua.blas.gemm.perform(
            PreferredType,
            cmd,
            alpha,
            null,
            a,
            .no_transpose,
            b,
            .transpose,
            beta,
            null,
            c,
        );
    }

    try c.wait();

    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(@divTrunc(end_ts - start_ts, niterations))) / 1000.0;

    return total_diff;
}

inline fn run_wekua_cpu_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    return run_wekua_test(.cpu, allocator, size, alphas, betas, buf1, buf2, buf3);
}

inline fn run_wekua_gpu_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    return run_wekua_test(.gpu, allocator, size, alphas, betas, buf1, buf2, buf3);
}

const tests = .{
    .{ "OpenBLAS", run_openblas_test },
    .{ "Wekua (C)", run_old_wekua_cpu_test },
    .{ "Wekua (C) GPU", run_old_wekua_gpu_test },
    .{ "wekua (Zig)", run_wekua_cpu_test },
    .{ "wekua (Zig) GPU", run_wekua_gpu_test },
};

pub fn run_benchmark(allocator: std.mem.Allocator) ![]utils.report {
    const reports: []utils.report = try allocator.alloc(utils.report, tests.len);
    errdefer allocator.free(reports);

    inline for (tests, 0..) |test_info, report_index| {
        const test_name = test_info[0];
        const test_func = test_info[1];

        var report = utils.report{ .name = test_name };

        var size: u64 = starting_point;
        for (&report.avg_times_per_batch) |*t| {
            const a = try allocator.alloc(PreferredType, size * size);
            defer allocator.free(a);

            const b = try allocator.alloc(PreferredType, size * size);
            defer allocator.free(b);

            const c = try allocator.alloc(PreferredType, size * size);
            defer allocator.free(c);

            const randprg = std.crypto.random;
            for (a, b, c) |*v1, *v2, *v3| {
                v1.* = randprg.float(PreferredType);
                v2.* = randprg.float(PreferredType);
                v3.* = randprg.float(PreferredType);
            }

            const alphas = try allocator.alloc(PreferredType, niterations);
            defer allocator.free(alphas);

            const betas = try allocator.alloc(PreferredType, niterations);
            defer allocator.free(betas);

            for (alphas, betas) |*alpha, *beta| {
                alpha.* = randprg.float(PreferredType);
                beta.* = randprg.float(PreferredType);
            }

            t.* = try test_func(
                allocator,
                size,
                alphas,
                betas,
                a,
                b,
                c,
            );

            size *= 2;
        }

        reports[report_index] = report;
    }

    return reports;
}
