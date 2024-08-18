const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");


fn test_fill_uniform(
    _: std.mem.Allocator, cmd: wekua.command_queue.wCommandQueue,
    tensor: wekua.tensor.wTensor
) !void {
    try wekua.tensor.extra.random.random(cmd, tensor);
}

test "Create float tensor with random numbers" {
    const allocator = std.testing.allocator;
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{100, 100, 100}, .{.dtype = wekua.tensor.wTensorDtype.float64});
    defer wekua.tensor.release(tensor);
    
    try std.testing.checkAllAllocationFailures(
        allocator, test_fill_uniform, .{ctx.command_queues[0], tensor}
    );
}

test "Check variance" {
    const allocator = std.testing.allocator;
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{100, 100, 100}, .{.dtype = wekua.tensor.wTensorDtype.float64});
    defer wekua.tensor.release(tensor);
    
    const cmd = ctx.command_queues[0];
    try wekua.tensor.extra.random.random(cmd, tensor);

    const buf = try allocator.alloc(f64, tensor.number_of_elements);
    defer allocator.free(buf);

    try wekua.tensor.extra.io.write_to_buffer(cmd, tensor, buf);

    var mean: f64 = 0.0;
    for (buf) |elem| mean += elem;

    const number_of_elements_float: f64 = @floatFromInt(tensor.number_of_elements);
    mean /= number_of_elements_float;

    var sq_diff: f64 = 0.0;
    for (buf) |elem| {
        const diff = mean - elem;
        sq_diff += diff*diff;
    }

    const variance: f64 = sq_diff / number_of_elements_float;

    const eps = std.math.floatEps(f64);
    try std.testing.expect((variance - 0.07) >= eps and (0.09 - variance) >= eps);
}
