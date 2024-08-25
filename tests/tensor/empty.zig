const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");


fn create_and_release(allocator: std.mem.Allocator, config: wekua.tensor.wCreateTensorConfig) !void {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const shape_expected: []const u64 = &[_]u64{20, 10};
    const tensor = try wekua.tensor.empty(ctx, shape_expected, config);
    defer wekua.tensor.release(tensor);

    if (config.is_complex) {
        try std.testing.expect(tensor.is_complex);
        try std.testing.expect(!tensor.vectors_enabled);
        try std.testing.expect(tensor.number_of_elements_without_pitch == 2*20*10);
        try std.testing.expect(tensor.row_pitch == tensor.row_pitch_for_vectors);
    }else{
        try std.testing.expect(!tensor.is_complex);
        try std.testing.expect(tensor.vectors_enabled);
        try std.testing.expect(tensor.number_of_elements_without_pitch == 20*10);
    }

    try std.testing.expectEqualSlices(u64, shape_expected, tensor.shape);
}

fn test_tensor_creation(allocator: std.mem.Allocator) !void {
    try create_and_release(allocator, .{
        .dtype = .float32
    });

    try create_and_release(allocator, .{
        .dtype = .float32,
        .is_complex = true
    });
}

test "create and release" {
    const allocator = std.testing.allocator;
    try test_tensor_creation(allocator);
}

test "create and fail" {
    const allocator = std.testing.allocator;
    try std.testing.checkAllAllocationFailures(allocator, test_tensor_creation, .{});
}
