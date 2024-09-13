const std = @import("std");
const wekua = @import("wekua");
const cl = wekua.opencl;
const utils = @import("utils.zig");

pub const name: []const u8 = "AXPY";

fn run_wekua_test(
    _: std.mem.Allocator, cmd: wekua.command_queue.wCommandQueue, x: wekua.tensor.wTensor, y: wekua.tensor.wTensor,
    randprg: std.Random
) !f64 {
    const alpha: wekua.tensor.wScalar = .{.float64 = randprg.float(f64)};
    try wekua.blas.axpy(cmd, x, alpha, null, y);

    // TODO: Discover where is getting stuck
    var total_diff: f64 = 0.0;
    for (0..100) |_| {
        try wekua.tensor.extra.random.random(cmd, x);
        try wekua.tensor.extra.random.random(cmd, y);

        try wekua.tensor.wait(x);
        try wekua.tensor.wait(y);

        const start_ts = std.time.microTimestamp();
        try wekua.blas.axpy(cmd, x, alpha, null, y);
        try wekua.tensor.wait(x);
        const end_ts = std.time.microTimestamp();

        total_diff += @as(f64, @floatFromInt(end_ts - start_ts)) / 1000.0;
    }

    return total_diff / 100.0;
}

const tests = .{
    .{"wekua", run_wekua_test}
};

pub fn run_benchmark(allocator: std.mem.Allocator) ![]utils.report {
    const reports: []utils.report = try allocator.alloc(utils.report, tests.len);
    errdefer allocator.free(reports);

    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.cpu);
    defer wekua.context.release(ctx);
    const randprg = std.crypto.random;
    const cmd = ctx.command_queues[0];

    inline for (tests, 0..) |test_info, report_index| {
        const test_name = test_info[0];
        const test_func = test_info[1];

        var report = utils.report{
            .name = test_name
        };

        var size: u64 = 64;
        for (&report.avg_times_per_bactch) |*t| {
            const tensor1 = try wekua.tensor.alloc(ctx, &.{size}, .{
                .dtype = .float64
            });
            defer wekua.tensor.release(tensor1);
            const tensor2 = try wekua.tensor.alloc(ctx, &.{size}, .{
                .dtype = .float64
            });
            defer wekua.tensor.release(tensor2);

            t.* = try test_func(allocator, cmd, tensor1, tensor2, randprg);
            size *= 2;
        }

        reports[report_index] = report;
    }

    return reports;
}
