const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn create_random_tensor(
    comptime T: type, ctx: *const wekua.core.Context, is_complex: bool, randprg: std.Random,
    real_scalar: ?T, imag_scalar: ?T
) !*wekua.Tensor(T) {
    const shape: [3]u64 = .{
        randprg.intRangeAtMost(u64, 2, 500),
        randprg.intRangeAtMost(u64, 2, 500),
        randprg.intRangeAtMost(u64, 2, 500)
    };

    const tensor = try wekua.Tensor(T).alloc(ctx, &shape, .{
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
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
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
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
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
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
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
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
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
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
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
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
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

test "create random buffer, write to tensor and read" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const randprg = std.crypto.random;
    const tensor = try create_random_tensor(ctx, false, randprg, null, null);
    defer wekua.tensor.release(tensor);

    const buf1: []i64 = try allocator.alloc(i64, tensor.number_of_elements_without_padding);
    defer allocator.free(buf1);

    const buf2: []i64 = try allocator.alloc(i64, tensor.number_of_elements_without_padding);
    defer allocator.free(buf2);

    for (buf1) |*element| {
        element.* = randprg.intRangeAtMost(i64, -1000, 1000);
    }

    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.io.read_from_buffer(
        w_cmd, tensor, buf1
    );

    try wekua.tensor.io.write_to_buffer(
        w_cmd, tensor, buf2
    );

    for (buf1, buf2) |e1, e2| {
        try std.testing.expectEqual(e1, e2);
    }
}

fn copy_tensors_and_check(
    _: std.mem.Allocator, src: wekua.tensor.wTensor, dst: wekua.tensor.wTensor,
    command_queue: wekua.command_queue.wCommandQueue, buf1: []f64, buf2: []f64
) !void {
    try wekua.tensor.io.copy(command_queue, src, dst);

    try wekua.tensor.io.write_to_buffer(command_queue, src, buf1);
    try wekua.tensor.io.write_to_buffer(command_queue, dst, buf2);

    const eps = std.math.floatEps(f64);
    for (buf1, buf2) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, eps);
    }
}

test "Copy tensors with same row pitch" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const cmd = ctx.command_queues[0];
    const tensor1 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64});
    defer wekua.tensor.release(tensor1);

    const tensor2 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64});
    defer wekua.tensor.release(tensor2);

    try wekua.tensor.extra.random.random(cmd, tensor1);

    const buf1: []f64 = try allocator.alloc(f64, tensor1.number_of_elements_without_padding);
    defer allocator.free(buf1);

    const buf2: []f64 = try allocator.alloc(f64, tensor1.number_of_elements_without_padding);
    defer allocator.free(buf2);

    try std.testing.checkAllAllocationFailures(allocator, copy_tensors_and_check, .{
        tensor1, tensor2, cmd, buf1, buf2
    });
}

test "Copy tensors with different row pitch" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const cmd = ctx.command_queues[0];
    const tensor1 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64});
    defer wekua.tensor.release(tensor1);

    const tensor2 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64, .vectors_enabled = false});
    defer wekua.tensor.release(tensor2);

    try wekua.tensor.extra.random.random(cmd, tensor1);

    const buf1: []f64 = try allocator.alloc(f64, tensor1.number_of_elements_without_padding);
    defer allocator.free(buf1);

    const buf2: []f64 = try allocator.alloc(f64, tensor1.number_of_elements_without_padding);
    defer allocator.free(buf2);

    try std.testing.checkAllAllocationFailures(allocator, copy_tensors_and_check, .{
        tensor1, tensor2, cmd, buf1, buf2
    });
}

test "Try to copy tensors with different dtype" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const cmd = ctx.command_queues[0];
    const tensor1 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64});
    defer wekua.tensor.release(tensor1);

    const tensor2 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .int8});
    defer wekua.tensor.release(tensor2);

    const ret = wekua.tensor.io.copy(cmd, tensor1, tensor2);
    try std.testing.expectError(error.UnqualTensorsDtype, ret);
}

test "Try to copy tensors with different shape" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const cmd = ctx.command_queues[0];
    const tensor1 = try wekua.tensor.empty(ctx, &.{100, 50}, .{.dtype = .float64});
    defer wekua.tensor.release(tensor1);

    const tensor2 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64});
    defer wekua.tensor.release(tensor2);

    const ret = wekua.tensor.io.copy(cmd, tensor1, tensor2);
    try std.testing.expectError(error.UnqualTensorsShape, ret);
}

test "Try to copy complex and real tensors" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const cmd = ctx.command_queues[0];
    const tensor1 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64, .is_complex = true});
    defer wekua.tensor.release(tensor1);

    const tensor2 = try wekua.tensor.empty(ctx, &.{100, 100}, .{.dtype = .float64});
    defer wekua.tensor.release(tensor2);

    const ret = wekua.tensor.io.copy(cmd, tensor1, tensor2);
    try std.testing.expectError(error.TensorIsnotComplex, ret);
}
