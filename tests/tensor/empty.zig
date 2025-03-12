const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");


fn create_and_release(comptime T: type, allocator: std.mem.Allocator, config: wekua.CreateTensorConfig) !void {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const shape_expected: []const u64 = &[_]u64{20, 10};

    const tensor = try wekua.Tensor(T).empty(ctx, shape_expected, config);
    defer tensor.release();

    if (config.is_complex) {
        try std.testing.expect(tensor.is_complex);
        try std.testing.expect(!tensor.vectors_enabled);
        try std.testing.expectEqual(2*20*10, tensor.number_of_elements_without_padding);
        try std.testing.expectEqual(tensor.row_pitch, tensor.row_pitch_for_vectors);
        try std.testing.expectEqual(2, tensor.pitchs[tensor.pitchs.len - 1]);
    }else{
        try std.testing.expect(!tensor.is_complex);
        try std.testing.expect(tensor.vectors_enabled);
        try std.testing.expectEqual(20*10, tensor.number_of_elements_without_padding);
        try std.testing.expectEqual(1, tensor.pitchs[tensor.pitchs.len - 1]);
    }

    try std.testing.expectEqualSlices(u64, shape_expected, tensor.shape);
}

fn test_tensor_creation(allocator: std.mem.Allocator) !void {
    inline for (wekua.tensor.SupportedTypes) |T| {
        try create_and_release(T, allocator, .{});

        try create_and_release(T, allocator, .{
            .is_complex = true
        });
    }
}

test "create and release" {
    const allocator = std.testing.allocator;
    try test_tensor_creation(allocator);
}

test "create and fail" {
    const allocator = std.testing.allocator;
    try std.testing.checkAllAllocationFailures(allocator, test_tensor_creation, .{});
}
