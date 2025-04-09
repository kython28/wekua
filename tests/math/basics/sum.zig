const wekua = @import("wekua");
const cl = wekua.opencl;
const std = @import("std");

fn test_sum(
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

    try wekua.tensor.random.uniform(T, command_queue, tensor, null, null, null);

    const allocator = context.allocator;
    const buffer1 = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(buffer1);

    try wekua.tensor.memory.writeToBuffer(T, tensor, command_queue, buffer1);

    const eps = 0.1;
    var actual_result: T = undefined;
    if (is_complex) {
        var actual_result_complex: T = undefined;
        try wekua.math.basic.sum(T, command_queue, tensor, &actual_result, &actual_result_complex);

        var expected_result: T = 0;
        var expected_result_complex: T = 0;
        for (0..(tensor.dimensions.number_of_elements_without_padding / 2)) |i| {
            const index = i * 2;

            expected_result += buffer1[index];
            expected_result_complex += buffer1[index + 1];
        }

        try std.testing.expectApproxEqAbs(expected_result, actual_result, eps);
        try std.testing.expectApproxEqAbs(expected_result_complex, actual_result_complex, eps);
    }else{
        try wekua.math.basic.sum(T, command_queue, tensor, &actual_result, null);

        var expected_result: T = 0;
        for (buffer1) |v| {
            expected_result += v;
        }

        try std.testing.expectApproxEqAbs(expected_result, actual_result, eps);
    }
}

test "Sum" {
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
                try test_sum(T, ctx, command_queue, is_complex, vector_enabled);
            }
        }
    }
}
