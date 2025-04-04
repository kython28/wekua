const std = @import("std");
const builtin = @import("builtin");

const wekua = @import("wekua");
const cl = wekua.opencl;
const utils = @import("utils.zig");

pub const name: []const u8 = "AXPY";
pub const starting_point: u64 = switch (builtin.mode) {
    .Debug => 4,
    else => 4096,
};

const niterations = switch (builtin.mode) {
    .Debug => 100,
    else => 1000,
};

const PreferredType = f32;
const WekuaCPreferredType = utils.wekua_c.WEKUA_DTYPE_FLOAT;

fn run_openblas_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    buf1: []PreferredType,
    buf2: []PreferredType,
) !f64 {
    const axpy_func = if (PreferredType == f32) utils.openblas.cblas_saxpy else utils.openblas.cblas_daxpy;

    const x = try allocator.dupe(PreferredType, buf1);
    defer allocator.free(x);

    const y = try allocator.dupe(PreferredType, buf2);
    defer allocator.free(y);

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (0..niterations, alphas) |i, alpha| {
        if (i % 2 == 0) {
            axpy_func(@intCast(size), alpha, x.ptr, 1, y.ptr, 1);
        } else {
            axpy_func(@intCast(size), alpha, y.ptr, 1, x.ptr, 1);
        }
    }
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(end_ts - start_ts)) / 1000.0;

    return total_diff;
}

fn run_old_wekua_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    x_buf: []PreferredType,
    y_buf: []PreferredType,
) !f64 {
    const ctx: utils.wekua_c.wekuaContext = utils.wekua_c.createSomeWekuaContext(
        @intFromEnum(cl.device.enums.device_type.all),
        1,
        0,
    ) orelse return error.OutOfMemory;
    defer utils.wekua_c.freeWekuaContext(ctx);

    const x = utils.wekua_c.wekuaAllocMatrix(ctx, 1, size, WekuaCPreferredType) orelse return error.OutOfMemory;
    defer utils.wekua_c.wekuaFreeMatrix(x, 0, null);

    const y = utils.wekua_c.wekuaAllocMatrix(ctx, 1, size, WekuaCPreferredType) orelse return error.OutOfMemory;
    defer utils.wekua_c.wekuaFreeMatrix(y, 0, null);

    var ret = utils.wekua_c.wekuaMatrixCopyBuffer(x, x_buf.ptr, null);
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);

    ret = utils.wekua_c.wekuaMatrixCopyBuffer(y, y_buf.ptr, null);
    if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);

    _ = utils.wekua_c.compileKernel(ctx, utils.wekua_c.WEKUA_KERNEL_AXPY, WekuaCPreferredType, 0);

    const events: []cl.event.cl_event = try allocator.alloc(cl.event.cl_event, niterations);
    defer allocator.free(events);

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (0..niterations, alphas) |i, alpha| {
        const prev_event: ?*cl.event.cl_event = if (i > 0) &events[i - 1] else null;

        if (i % 2 == 0) {
            ret = utils.wekua_c.wekuaBlasAxpy(
                x,
                y,
                @constCast(&alpha),
                null,
                @as(u32, @intFromBool(i > 0)),
                @ptrCast(prev_event),
                @ptrCast(&events[i]),
            );
            if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);
        } else {
            ret = utils.wekua_c.wekuaBlasAxpy(
                y,
                x,
                @constCast(&alpha),
                null,
                @as(u32, @intFromBool(i > 0)),
                @ptrCast(prev_event),
                @ptrCast(&events[i]),
            );
            if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);
        }
    }
    try cl.event.wait(events[niterations - 1]);
    for (events) |event| {
        cl.event.release(event);
    }
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(end_ts - start_ts)) / 1000.0;

    return total_diff;
}

fn run_wekua_test(
    allocator: std.mem.Allocator,
    size: usize,
    alphas: []PreferredType,
    x_buf: []PreferredType,
    y_buf: []PreferredType,
) !f64 {
    const ctx = try wekua.core.Context.create_from_best_device(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const cmd = &ctx.command_queues[0];

    const FloatTensor = wekua.Tensor(PreferredType);

    const x = try FloatTensor.alloc(ctx, &.{size}, .{});
    defer x.release();

    const y = try FloatTensor.alloc(ctx, &.{size}, .{});
    defer y.release();

    // At the first time, we need to compile the kernels
    try wekua.blas.axpy(PreferredType, cmd, x, 0, null, y);

    try wekua.tensor.memory.readFromBuffer(PreferredType, x, cmd, x_buf);
    try wekua.tensor.memory.readFromBuffer(PreferredType, y, cmd, y_buf);

    try x.wait();
    try y.wait();

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (0..niterations, alphas) |i, alpha| {
        if (i % 2 == 0) {
            try wekua.blas.axpy(PreferredType, cmd, x, alpha, null, y);
        } else {
            try wekua.blas.axpy(PreferredType, cmd, y, alpha, null, x);
        }
    }

    try x.wait();
    try y.wait();

    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(end_ts - start_ts)) / 1000.0;

    return total_diff;
}

const tests = .{
    .{ "OpenBLAS", run_openblas_test },
    .{ "Wekua (C)", run_old_wekua_test },
    .{ "wekua (Zig)", run_wekua_test },
};

pub fn run_benchmark(allocator: std.mem.Allocator) ![]utils.report {
    const reports: []utils.report = try allocator.alloc(utils.report, tests.len);
    errdefer allocator.free(reports);

    inline for (tests, 0..) |test_info, report_index| {
        const test_name = test_info[0];
        const test_func = test_info[1];

        var report = utils.report{ .name = test_name };

        var size: u64 = starting_point;
        for (&report.avg_times_per_bactch) |*t| {
            const x = try allocator.alloc(PreferredType, size);
            defer allocator.free(x);

            const y = try allocator.alloc(PreferredType, size);
            defer allocator.free(y);

            const randprg = std.crypto.random;
            for (x, y) |*v1, *v2| {
                v1.* = randprg.float(PreferredType);
                v2.* = randprg.float(PreferredType);
            }

            const alphas = try allocator.alloc(PreferredType, niterations);
            defer allocator.free(alphas);

            for (alphas) |*alpha| {
                alpha.* = -10 + randprg.float(PreferredType) * 20.0;
            }

            t.* = try test_func(allocator, size, alphas, x, y);
            size *= 2;
        }

        reports[report_index] = report;
    }

    return reports;
}
