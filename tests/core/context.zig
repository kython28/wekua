const std = @import("std");
const cl = @import("opencl");

const wekua = @import("wekua");

fn testCreateContext(allocator: std.mem.Allocator) !void {
    const ctx = try wekua.core.Context.initFromDeviceType(
        allocator,
        null,
        cl.device.Type.all,
    );
    ctx.release();
}

fn testCreateContextWithBestDevice(allocator: std.mem.Allocator) !void {
    const ctx = try wekua.core.Context.initFromBestDevice(
        allocator,
        null,
        cl.device.Type.all,
    );
    ctx.release();
}

test "create" {
    try testCreateContext(std.testing.allocator);
    try testCreateContextWithBestDevice(std.testing.allocator);
}

test "create_with_fail" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, testCreateContext, .{});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, testCreateContextWithBestDevice, .{});
}
