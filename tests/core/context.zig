const std = @import("std");
const cl = @import("opencl");

const wekua = @import("wekua");

fn test_create_context(allocator: std.mem.Allocator) !void {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    ctx.release();
}

fn test_create_context_with_best_device(allocator: std.mem.Allocator) !void {
    const ctx = try wekua.core.Context.create_from_best_device(allocator, null, cl.device.enums.device_type.all);
    ctx.release();
}

test "create" {
    try test_create_context(std.testing.allocator);
    try test_create_context_with_best_device(std.testing.allocator);
}

test "create_with_fail" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, test_create_context, .{});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, test_create_context_with_best_device, .{});
}
