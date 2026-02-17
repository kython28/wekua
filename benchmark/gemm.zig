const std = @import("std");
const builtin = @import("builtin");

const wekua = @import("wekua");
const cl = wekua.opencl;
const utils = @import("utils.zig");

pub const name: []const u8 = "GEMM";
pub const dtype: []const u8 = "f32";
pub const starting_point: u64 = 4;

pub const niterations: u64 = switch (builtin.mode) {
    .Debug => 1,
    else => 1,
};

const PreferredType = f32;
const WekuaCPreferredType = utils.wekua_c.WEKUA_DTYPE_FLOAT;
const GemmOperation = wekua.blas.GemmOperation;

fn toCblasTranspose(op: GemmOperation) c_uint {
    return switch (op) {
        .no_transpose => utils.openblas.CblasNoTrans,
        .transpose => utils.openblas.CblasTrans,
    };
}

fn opName(op: GemmOperation) []const u8 {
    return switch (op) {
        .no_transpose => "N",
        .transpose => "T",
    };
}

fn run_openblas_test(
    op_a: GemmOperation,
    op_b: GemmOperation,
    _: cl.device.Type,
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
            toCblasTranspose(op_a),
            toCblasTranspose(op_b),
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
    total_diff += @as(f64, @floatFromInt(@divTrunc(end_ts - start_ts, @as(i64, @intCast(niterations))))) / 1000.0;

    return total_diff;
}

fn run_old_wekua_test(
    op_a: GemmOperation,
    op_b: GemmOperation,
    device_type: cl.device.Type,
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
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translateOpenCLError(ret);

    ret = utils.wekua_c.wekuaMatrixCopyBuffer(b, buf2.ptr, null);
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translateOpenCLError(ret);

    ret = utils.wekua_c.wekuaMatrixCopyBuffer(c, buf3.ptr, null);
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translateOpenCLError(ret);

    _ = utils.wekua_c.compileKernel(ctx, utils.wekua_c.WEKUA_KERNEL_GEMM, WekuaCPreferredType, 0);

    const events: []cl.event.Event = try allocator.alloc(cl.event.Event, niterations);
    defer allocator.free(events);

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (alphas, betas) |alpha, beta| {
        ret = utils.wekua_c.wekuaBlasGemm(
            @constCast(&alpha),
            null,
            @intFromEnum(op_a),
            a,
            @intFromEnum(op_b),
            b,
            @constCast(&beta),
            null,
            c,
            0,
            null,
        );
        if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translateOpenCLError(ret);
    }
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(@divTrunc(end_ts - start_ts, @as(i64, @intCast(niterations))))) / 1000.0;

    return total_diff;
}

fn run_wekua_test(
    op_a: GemmOperation,
    op_b: GemmOperation,
    device_type: cl.device.Type,
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    betas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
    buf3: []PreferredType,
) !f64 {
    const context = try wekua.core.Context.initFromBestDevice(allocator, null, device_type);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try wekua.core.Pipeline.init(command_queue);
    defer pipeline.deinit();

    try pipeline.prealloc(alphas.len);

    const FloatTensor = wekua.Tensor(PreferredType);

    const a = try FloatTensor.alloc(context, pipeline, &.{ size, size }, .{});
    defer a.release(pipeline);

    const b = try FloatTensor.alloc(context, pipeline, &.{ size, size }, .{});
    defer b.release(pipeline);

    const c = try FloatTensor.alloc(context, pipeline, &.{ size, size }, .{});
    defer c.release(pipeline);

    try wekua.tensor_module.memory.readFromBuffer(PreferredType, pipeline, a, buf1);
    try wekua.tensor_module.memory.readFromBuffer(PreferredType, pipeline, b, buf2);
    try wekua.tensor_module.memory.readFromBuffer(PreferredType, pipeline, c, buf3);

    const packed_tensors = try wekua.blas.GemmPackedTensors(PreferredType).init(pipeline, c, size, true);
    defer packed_tensors.deinit(pipeline);

    // Warmup: compile kernels
    try wekua.blas.gemm(PreferredType, pipeline, 1.0, a, op_a, b, op_b, 0.0, c, packed_tensors);
    pipeline.waitAndCleanup();

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (alphas, betas) |alpha, beta| {
        try wekua.blas.gemm(PreferredType, pipeline, alpha, a, op_a, b, op_b, beta, c, packed_tensors);
    }
    pipeline.waitAndCleanup();

    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(@divTrunc(end_ts - start_ts, @as(i64, @intCast(niterations))))) / 1000.0;

    return total_diff;
}

const operations = .{
    .{ GemmOperation.no_transpose, GemmOperation.no_transpose },
    .{ GemmOperation.no_transpose, GemmOperation.transpose },
    .{ GemmOperation.transpose, GemmOperation.no_transpose },
    .{ GemmOperation.transpose, GemmOperation.transpose },
};

const backends = .{
    .{ "OpenBLAS", cl.device.Type.cpu, run_openblas_test },
    // .{ "Wekua (C)", cl.device.Type.cpu, run_old_wekua_test },
    // .{ "Wekua (C) GPU", cl.device.Type.gpu, run_old_wekua_test },
    .{ "wekua (Zig)", cl.device.Type.cpu, run_wekua_test },
    // .{ "wekua (Zig) GPU", cl.device.Type.gpu, run_wekua_test },
};

const num_tests = operations.len * backends.len;

pub fn run_benchmark(allocator: std.mem.Allocator) ![]utils.report {
    const reports: []utils.report = try allocator.alloc(utils.report, num_tests);
    errdefer allocator.free(reports);

    var report_index: usize = 0;
    inline for (operations) |op_pair| {
        const op_a = op_pair[0];
        const op_b = op_pair[1];

        inline for (backends) |backend| {
            const backend_name = backend[0];
            const device_type = backend[1];
            const test_func = backend[2];

            const test_name = backend_name ++ " (" ++ (comptime opName(op_a)) ++ (comptime opName(op_b)) ++ ")";
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
                    op_a,
                    op_b,
                    device_type,
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
            report_index += 1;
        }
    }

    return reports;
}
