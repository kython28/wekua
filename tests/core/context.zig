const std = @import("std");
const cl = @import("opencl");

const wekua = @import("wekua");

fn test_create_context(allocator: std.mem.Allocator) !void {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    wekua.context.release(ctx);
}

test "create" {
    try test_create_context(std.testing.allocator);
}

test "create_with_fail" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, test_create_context, .{});
}
