const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn create_and_release(config: wekua.tensor.wCreateTensorConfig) !void {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.empty(ctx, &[_]u64{20, 10}, config);
    defer wekua.tensor.release(tensor);

    if (config.is_complex) {
        try std.testing.expect(tensor.is_complex);
        try std.testing.expect(!tensor.vectors_enabled);
        try std.testing.expect(tensor.number_of_elements == 2*20*10);
        try std.testing.expect(tensor.row_pitch == tensor.row_pitch_for_vectors);
    }else{
        try std.testing.expect(!tensor.is_complex);
        try std.testing.expect(tensor.vectors_enabled);
    }
}

test "create and release" {
    try create_and_release(.{
        .dtype = .float32
    });

    try create_and_release(.{
        .dtype = .float32,
        .is_complex = true
    });
}

test "create and fail" {
    const max_index = 20;
    for (0..max_index) |i| {
        var failing_allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{
            .fail_index = i
        });
        const ally = failing_allocator.allocator();

        const context = wekua.context.create_from_device_type(
            ally, null, cl.device.enums.device_type.all
        ) catch |err| switch (err) {
            error.OutOfMemory => return,
            else => return err
        };
        defer wekua.context.release(context);

        const tensor = wekua.tensor.empty(context, &[_]u64{100, 20, 100}, .{
            .dtype = wekua.tensor.wTensorDtype.uint32
        }) catch |err| switch (err) {
            error.OutOfMemory => return,
            else => return err
    };
        wekua.tensor.release(tensor);
    }
}
