const std = @import("std");
const wekua = @import("wekua");
const cl = wekua.opencl;
const utils = @import("utils.zig");

pub const name: []const u8 = "AXPY";
pub const starting_point: u64 = 4096;

const niterations = 1000;

fn run_openblas_test(allocator: std.mem.Allocator, size: usize) !f64 {
    const x: []f64 = try allocator.alloc(f64, size);
    defer allocator.free(x);

    const y: []f64 = try allocator.alloc(f64, size);
    defer allocator.free(y);

    const randprg = std.crypto.random;
    for (x, y) |*v1, *v2| {
        v1.* = randprg.float(f64);
        v2.* = randprg.float(f64);
    }

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (0..niterations) |i| {
        const alpha = -10 + randprg.float(f64)*20.0;

        if (i%2 == 0) {
            utils.openblas.cblas_daxpy(@intCast(size), alpha, x.ptr, 1, y.ptr, 1);
        }else{
            utils.openblas.cblas_daxpy(@intCast(size), alpha, y.ptr, 1, x.ptr, 1);
        }
    }
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(end_ts - start_ts)) / 1000.0;

    return total_diff;
}

fn run_old_wekua_test(allocator: std.mem.Allocator, size: usize) !f64 {
    const ctx: utils.wekua_c.wekuaContext = utils.wekua_c.createSomeWekuaContext(@intFromEnum(cl.device.enums.device_type.cpu), 1, 0)
        orelse return error.OutOfMemory;
    defer utils.wekua_c.freeWekuaContext(ctx);

    const randprg = std.crypto.random;

    const x = utils.wekua_c.wekuaMatrixRandn(ctx, 1, size, 0) orelse return error.OutOfMemory;
    defer utils.wekua_c.wekuaFreeMatrix(x, 0, null);
    const y = utils.wekua_c.wekuaMatrixRandn(ctx, 1, size, 0) orelse return error.OutOfMemory;
    defer utils.wekua_c.wekuaFreeMatrix(y, 0, null);

    _ = utils.wekua_c.compileKernel(ctx, utils.wekua_c.WEKUA_KERNEL_AXPY, utils.wekua_c.WEKUA_DTYPE_DOUBLE, 0);

    const events: []cl.event.cl_event = try allocator.alloc(cl.event.cl_event, niterations);
    defer allocator.free(events);

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (0..niterations) |i| {
        const alpha = -10 + randprg.float(f64)*20.0;
        const prev_event: ?*cl.event.cl_event = if (i > 0) &events[i - 1] else null;

        if (i%2 == 0) {
            const ret = utils.wekua_c.wekuaBlasAxpy(
                x, y, @constCast(&alpha), null, @as(u32, @intFromBool(i > 0)), @ptrCast(prev_event), @ptrCast(&events[i])
            );
            if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);
        }else{
            const ret = utils.wekua_c.wekuaBlasAxpy(
                y, x, @constCast(&alpha), null, @as(u32, @intFromBool(i > 0)), @ptrCast(prev_event), @ptrCast(&events[i])
            );
            if (ret != utils.wekua_c.CL_SUCCESS) return cl.errors.translate_opencl_error_for_all_fields(ret);
        }
    }
    try cl.event.wait(events[niterations - 1]);
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(end_ts - start_ts)) / 1000.0;

    return total_diff;
}

fn run_wekua_test(allocator: std.mem.Allocator, size: usize) !f64 {
    const ctx = try wekua.context.create_from_best_device(allocator, null, cl.device.enums.device_type.cpu);
    defer wekua.context.release(ctx);
    const randprg = std.crypto.random;
    const cmd = ctx.command_queues[0];

    const x = try wekua.tensor.alloc(ctx, &.{size}, .{
        .dtype = .float64
    });
    defer wekua.tensor.release(x);
    const y = try wekua.tensor.alloc(ctx, &.{size}, .{
        .dtype = .float64
    });
    defer wekua.tensor.release(y);

    var alpha: wekua.tensor.wScalar = .{.float64 = randprg.float(f64)};
    try wekua.tensor.extra.random.random(cmd, x);
    try wekua.tensor.extra.random.random(cmd, y);
    try wekua.blas.axpy(cmd, x, alpha, null, y);
    try wekua.tensor.wait(x);
    try wekua.tensor.wait(y);

    var total_diff: f64 = 0.0;
    const start_ts = std.time.microTimestamp();
    for (0..niterations) |i| {
        alpha.float64 = -10.0 + randprg.float(f64)*20.0;
        if (i%2 == 0) {
            try wekua.blas.axpy(cmd, x, alpha, null, y);
        }else{
            try wekua.blas.axpy(cmd, y, alpha, null, x);
        }
    }
    try wekua.tensor.wait(x);
    try wekua.tensor.wait(y);
    const end_ts = std.time.microTimestamp();
    total_diff += @as(f64, @floatFromInt(end_ts - start_ts)) / 1000.0;

    return total_diff;
}

const tests = .{
    .{"OpenBLAS", run_openblas_test},
    .{"Wekua (C)", run_old_wekua_test},
    .{"wekua (Zig)", run_wekua_test}
};

pub fn run_benchmark(allocator: std.mem.Allocator) ![]utils.report {
    const reports: []utils.report = try allocator.alloc(utils.report, tests.len);
    errdefer allocator.free(reports);

    inline for (tests, 0..) |test_info, report_index| {
        const test_name = test_info[0];
        const test_func = test_info[1];

        var report = utils.report{
            .name = test_name
        };

        var size: u64 = starting_point;
        for (&report.avg_times_per_bactch) |*t| {
            t.* = try test_func(allocator, size);
            size *= 2;
        }

        reports[report_index] = report;
    }

    return reports;
}
