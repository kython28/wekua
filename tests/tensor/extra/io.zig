const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

test "fill and get random value" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const shape: [3]u64 = .{
        randprg.intRangeAtMost(u64, 1, 500),
        randprg.intRangeAtMost(u64, 1, 500),
        randprg.intRangeAtMost(u64, 1, 500)
    };

    const tensor = try wekua.tensor.alloc(ctx, &shape, .{.dtype = wekua.tensor.wTensorDtype.int64});
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const random_number: i64 = randprg.intRangeAtMost(i64, -1000, 1000);
    const scalar: wekua.tensor.wScalar = .{ .int64 = random_number };
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, null);

    var new_scalar: wekua.tensor.wScalar = undefined;
    const coor: [3]u64 = .{
        randprg.intRangeLessThan(u64, 1, shape[0]),
        randprg.intRangeLessThan(u64, 1, shape[1]),
        randprg.intRangeLessThan(u64, 1, shape[2])
    };
    try wekua.tensor.extra.io.get_value(w_cmd, tensor, &coor, &new_scalar, null);

    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_scalar) == .int64);
    try std.testing.expectEqual(scalar.int64, new_scalar.int64);
}
