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
    errdefer tensor.release();

    const w_cmd = &ctx.command_queues[0];
    if (real_scalar != null or imag_scalar != null) {
        try wekua.tensor.fill.constant(T, tensor, w_cmd, real_scalar, imag_scalar);
    }

    return tensor;
}

test "fill and get random value" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        const scalar = switch (@typeInfo(T)) {
            .int => randprg.int(T),
            .float => randprg.float(T),
            else => unreachable
        };

        const scalar2 = switch (@typeInfo(T)) {
            .int => randprg.int(T),
            .float => randprg.float(T),
            else => unreachable
        };

        const tensor = try create_random_tensor(T, ctx, false, randprg, scalar, null);
        defer tensor.release();

        var actual_scalar: T = undefined;

        const coor: [3]u64 = .{
            randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
            randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
            randprg.intRangeLessThan(u64, 1, tensor.shape[2])
        };

        const coor2: [3]u64 = .{
            randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
            randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
            randprg.intRangeLessThan(u64, 1, tensor.shape[2])
        };

        const w_cmd = &ctx.command_queues[0];
        try wekua.tensor.memory.putValue(T, tensor, w_cmd, &coor2, scalar2, null);
        try wekua.tensor.memory.getValue(T, tensor, w_cmd, &coor, &actual_scalar, null);

        switch (@typeInfo(T)) {
            .int => try std.testing.expectEqual(scalar, actual_scalar),
            .float => try std.testing.expectApproxEqAbs(scalar, actual_scalar, comptime std.math.floatEps(T)),
            else => unreachable
        }

        try wekua.tensor.memory.getValue(T, tensor, w_cmd, &coor2, &actual_scalar, null);
        switch (@typeInfo(T)) {
            .int => try std.testing.expectEqual(scalar2, actual_scalar),
            .float => try std.testing.expectApproxEqAbs(scalar2, actual_scalar, comptime std.math.floatEps(T)),
            else => unreachable
        }
    }
}

test "fill and get random complex value" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        const scalar = switch (@typeInfo(T)) {
            .int => randprg.int(T),
            .float => randprg.float(T),
            else => unreachable
        };
        const scalar_imag = switch (@typeInfo(T)) {
            .int => randprg.int(T),
            .float => randprg.float(T),
            else => unreachable
        };

        const tensor = try create_random_tensor(T, ctx, true, randprg, scalar, scalar_imag);
        defer tensor.release();

        var new_scalar: T = undefined;
        var new_scalar_imag: T = undefined;

        const coor: [3]u64 = .{
            randprg.intRangeLessThan(u64, 1, tensor.shape[0]),
            randprg.intRangeLessThan(u64, 1, tensor.shape[1]),
            randprg.intRangeLessThan(u64, 1, tensor.shape[2])
        };

        const w_cmd = &ctx.command_queues[0];
        try wekua.tensor.memory.getValue(T, tensor, w_cmd, &coor, &new_scalar, &new_scalar_imag);

        switch (@typeInfo(T)) {
            .int => {
                try std.testing.expectEqual(scalar, new_scalar);
                try std.testing.expectEqual(scalar_imag, new_scalar_imag);
            },
            .float => {
                try std.testing.expectApproxEqAbs(scalar, new_scalar, comptime std.math.floatEps(T));
                try std.testing.expectApproxEqAbs(scalar_imag, new_scalar_imag, comptime std.math.floatEps(T));
            },
            else => unreachable
        }
    }
}

test "read from and write to buffer" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        const tensor = try wekua.Tensor(T).empty(ctx, &.{5, 5}, .{});
        defer tensor.release();

        var buffer1: [25]T = undefined;
        var buffer2: [25]T = undefined;
        for (&buffer1) |*v| {
            v.* = switch (@typeInfo(T)) {
                .int => randprg.int(T),
                .float => randprg.float(T),
                else => unreachable
            };
        }

        const w_cmd = &ctx.command_queues[0];
        try wekua.tensor.memory.readFromBuffer(T, tensor, w_cmd, &buffer1);
        try wekua.tensor.memory.writeToBuffer(T, tensor, w_cmd, &buffer2);

        for (&buffer1, &buffer2) |v1, v2| {
            try std.testing.expectEqual(v1, v2);
        }
    }
}

test "copy" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        const tensor = try wekua.Tensor(T).empty(ctx, &.{5, 5}, .{});
        defer tensor.release();

        const tensor2 = try wekua.Tensor(T).empty(ctx, &.{5, 5}, .{});
        defer tensor2.release();

        var buffer1: [25]T = undefined;
        var buffer2: [25]T = undefined;
        for (&buffer1) |*v| {
            v.* = switch (@typeInfo(T)) {
                .int => randprg.int(T),
                .float => randprg.float(T),
                else => unreachable
            };
        }

        const w_cmd = &ctx.command_queues[0];
        try wekua.tensor.memory.readFromBuffer(T, tensor, w_cmd, &buffer1);
        try wekua.tensor.memory.copy(T, w_cmd, tensor, tensor2);
        try wekua.tensor.memory.writeToBuffer(T, tensor2, w_cmd, &buffer2);

        for (&buffer1, &buffer2) |v1, v2| {
            try std.testing.expectEqual(v1, v2);
        }
    }
}
