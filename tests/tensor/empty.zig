const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");


fn create_and_release(comptime T: type, ctx: *wekua.core.Context, config: wekua.CreateTensorConfig) !void {
    const shape_expected: []const u64 = &[_]u64{20, 10};

    const tensor = try wekua.Tensor(T).empty(ctx, shape_expected, config);
    defer tensor.release();

    if (config.is_complex) {
        try std.testing.expect(tensor.flags.is_complex);
        try std.testing.expect(!tensor.flags.vectors_enabled);
        try std.testing.expectEqual(2*20*10, tensor.dimensions.number_of_elements_without_padding);
        try std.testing.expectEqual(tensor.memory_layout.row_pitch, tensor.memory_layout.row_pitch_for_vectors);
        try std.testing.expectEqual(2, tensor.dimensions.pitches[tensor.dimensions.pitches.len - 1]);
    }else{
        try std.testing.expect(!tensor.flags.is_complex);
        try std.testing.expect(tensor.flags.vectors_enabled);
        try std.testing.expectEqual(20*10, tensor.dimensions.number_of_elements_without_padding);
        try std.testing.expectEqual(1, tensor.dimensions.pitches[tensor.dimensions.pitches.len - 1]);
    }

    try std.testing.expectEqualSlices(u64, shape_expected, tensor.dimensions.shape);
}

fn test_tensor_creation(allocator: std.mem.Allocator, ctx: *wekua.core.Context) !void {
    ctx.allocator = allocator;
    inline for (wekua.core.SupportedTypes) |T| {
        try create_and_release(T, ctx, .{});

        try create_and_release(T, ctx, .{
            .is_complex = true
        });
    }
}

test "create and release" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    try test_tensor_creation(allocator, ctx);
}

test "create and fail" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();
    defer ctx.allocator = allocator;

    try std.testing.checkAllAllocationFailures(allocator, test_tensor_creation, .{ctx});
}
