const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn create_random_tensor(
    ctx: wekua.context.wContext, is_complex: bool, randprg: std.Random,
    real_scalar: ?wekua.tensor.wScalar, imag_scalar: ?wekua.tensor.wScalar
) !wekua.tensor.wTensor {

    const shape: [3]u64 = .{
        randprg.intRangeAtMost(u64, 1, 500),
        randprg.intRangeAtMost(u64, 1, 500),
        randprg.intRangeAtMost(u64, 1, 500)
    };

    const tensor = try wekua.tensor.alloc(ctx, &shape, .{
        .dtype = wekua.tensor.wTensorDtype.int64,
        .is_complex = is_complex
    });
    errdefer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    if (real_scalar != null or imag_scalar != null) {
        try wekua.tensor.extra.fill(w_cmd, tensor, real_scalar, imag_scalar);
    }

    return tensor;
}

test "fill and get random value" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const real_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const tensor = try create_random_tensor(ctx, false, randprg, real_scalar, null);
    defer wekua.tensor.release(tensor);

    var new_scalar: wekua.tensor.wScalar = undefined;
    const coor: [3]u64 = .{
        randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[2])
    };
    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.io.get_value(w_cmd, tensor, &coor, &new_scalar, null);

    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_scalar) == .int64);
    try std.testing.expectEqual(real_scalar.int64, new_scalar.int64);
}

test "fill and get random complex value" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const real_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const tensor = try create_random_tensor(ctx, false, randprg, real_scalar, null);
    defer wekua.tensor.release(tensor);

    var new_scalar: wekua.tensor.wScalar = undefined;
    var new_imag_scalar: wekua.tensor.wScalar = undefined;
    const coor: [3]u64 = .{
        randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[2])
    };
    const w_cmd = ctx.command_queues[0];
    const ret = wekua.tensor.extra.io.get_value(w_cmd, tensor, &coor, &new_scalar, &new_imag_scalar);
    try std.testing.expectError(wekua.tensor.errors.TensorIsnotComplex, ret);
}

test "fill complex and get random value" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const real_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const imag_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const tensor = try create_random_tensor(ctx, true, randprg, real_scalar, imag_scalar);
    defer wekua.tensor.release(tensor);

    var new_real_scalar: wekua.tensor.wScalar = undefined;
    const coor: [3]u64 = .{
        randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[2])
    };
    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.io.get_value(w_cmd, tensor, &coor, &new_real_scalar, null);

    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_real_scalar) == .int64);
    try std.testing.expectEqual(real_scalar.int64, new_real_scalar.int64);
}

test "fill complex and get random complex value" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const real_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const imag_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const tensor = try create_random_tensor(ctx, true, randprg, real_scalar, imag_scalar);
    defer wekua.tensor.release(tensor);

    var new_real_scalar: wekua.tensor.wScalar = undefined;
    var new_imag_scalar: wekua.tensor.wScalar = undefined;
    const coor: [3]u64 = .{
        randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[2])
    };
    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.io.get_value(w_cmd, tensor, &coor, &new_real_scalar, &new_imag_scalar);

    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_real_scalar) == .int64);
    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_imag_scalar) == .int64);
    try std.testing.expectEqual(real_scalar.int64, new_real_scalar.int64);
    try std.testing.expectEqual(imag_scalar.int64, new_imag_scalar.int64);
}

test "fill, put random value and get" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const real_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const tensor = try create_random_tensor(ctx, false, randprg, null, null);
    defer wekua.tensor.release(tensor);

    var new_scalar: wekua.tensor.wScalar = undefined;
    const coor: [3]u64 = .{
        randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[2])
    };
    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.io.put_value(w_cmd, tensor, &coor, real_scalar, null);
    try wekua.tensor.extra.io.get_value(w_cmd, tensor, &coor, &new_scalar, null);

    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_scalar) == .int64);
    try std.testing.expectEqual(real_scalar.int64, new_scalar.int64);
}

test "fill complex, put complex random value and get" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const real_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const imag_scalar: wekua.tensor.wScalar = .{ .int64 = randprg.intRangeAtMost(i64, -1000, 1000) };
    const tensor = try create_random_tensor(ctx, true, randprg, null, null);
    defer wekua.tensor.release(tensor);

    var new_real_scalar: wekua.tensor.wScalar = undefined;
    var new_imag_scalar: wekua.tensor.wScalar = undefined;
    const coor: [3]u64 = .{
        randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
        randprg.intRangeLessThan(u64, 1, tensor.shape[2])
    };
    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.io.put_value(w_cmd, tensor, &coor, real_scalar, imag_scalar);
    try wekua.tensor.extra.io.get_value(w_cmd, tensor, &coor, &new_real_scalar, &new_imag_scalar);

    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_real_scalar) == .int64);
    try std.testing.expect(@as(wekua.tensor.wTensorDtype, new_imag_scalar) == .int64);
    try std.testing.expectEqual(real_scalar.int64, new_real_scalar.int64);
    try std.testing.expectEqual(imag_scalar.int64, new_imag_scalar.int64);
}
