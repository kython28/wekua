const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

test "Create float tensor with random numbers" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{100, 100, 100}, .{.dtype = wekua.tensor.wTensorDtype.float64});
    defer wekua.tensor.release(tensor);
    
    try wekua.tensor.extra.random.random(ctx.command_queues[0], tensor);
}
