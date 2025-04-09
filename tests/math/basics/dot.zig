const wekua = @import("wekua");
const cl = wekua.opencl;
const std = @import("std");

inline fn complex_mul(comptime T: type, re1: T, im1: T, re2: T, im2: T) struct { T, T } {
    const k1: T = re2 * (re1 + im1);
    const k2: T = re1 * (im2 - re2);
    const k3: T = im1 * (re2 + im2);
    return .{ k1 - k3, k1 + k2 };
}

fn test_dot_product(
    comptime T: type,
    context: *wekua.core.Context,
    command_queue: *const wekua.core.CommandQueue,
    comptime is_complex: bool,
    vector_enabled: bool,
) !void {
    if (!command_queue.typeIsSupported(T)) return;

    const tensor = try wekua.Tensor(T).alloc(context, &.{10, 10, 10}, .{
        .is_complex = is_complex,
        .vectors_enabled = vector_enabled,
    });
    defer tensor.release();

    const tensor2 = try wekua.Tensor(T).alloc(context, &.{10, 10, 10}, .{
        .is_complex = is_complex,
        .vectors_enabled = vector_enabled,
    });
    defer tensor2.release();

    try wekua.tensor.random.uniform(T, command_queue, tensor, null, null, null);
    try wekua.tensor.random.uniform(T, command_queue, tensor2, null, null, null);

    const allocator = context.allocator;
    const buffer1 = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(buffer1);

    const buffer2 = try allocator.alloc(T, tensor2.dimensions.number_of_elements_without_padding);
    defer allocator.free(buffer2);

    try wekua.tensor.memory.writeToBuffer(T, tensor, command_queue, buffer1);
    try wekua.tensor.memory.writeToBuffer(T, tensor2, command_queue, buffer2);

    try wekua.math.basic.dot(T, command_queue, tensor, tensor2);

    const buffer3 = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(buffer3);

    try wekua.tensor.memory.writeToBuffer(T, tensor, command_queue, buffer3);

    const eps = comptime std.math.floatEps(T);
    if (is_complex) {
        for (0..(tensor.dimensions.number_of_elements_without_padding / 2)) |i| {
            const index = i * 2;
            const expected = complex_mul(T, buffer1[index], buffer1[index + 1], buffer2[index], buffer2[index + 1]);
            try std.testing.expectApproxEqAbs(expected[0], buffer3[index], eps);
            try std.testing.expectApproxEqAbs(expected[1], buffer3[index + 1], eps);
        }

    }else{
        for (buffer1, buffer2, buffer3) |a, b, c| {
            try std.testing.expectApproxEqAbs(a * b, c, eps);
        }
    }
}

test "Dot product" {
    const ctx = try wekua.core.Context.create_from_best_device(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.all,
    );
    defer ctx.release();

    const command_queue = &ctx.command_queues[0];

    inline for (&.{f32, f64}) |T| {
        inline for (&.{true, false}) |is_complex| {
            inline for (&.{true, false}) |vector_enabled| {
                try test_dot_product(T, ctx, command_queue, is_complex, vector_enabled);
            }
        }
    }
}
