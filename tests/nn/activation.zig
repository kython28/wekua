const std = @import("std");
const wekua = @import("wekua");
const cl = @import("opencl");

const core = wekua.core;
const CommandQueue = core.CommandQueue;

fn test_sigmoid(
    comptime T: type,
    allocator: std.mem.Allocator,
    ctx: *const wekua.core.Context,
    is_complex: bool,
    vectors_enabled: bool,
) !void {
    const cmd = &ctx.command_queues[0];
    if (!cmd.typeIsSupported(T)) {
        return;
    }

    const tensor = try wekua.Tensor(T).alloc(ctx, &.{ 10, 10 }, .{
        .is_complex = is_complex,
        .vectors_enabled = vectors_enabled
    });
    defer tensor.release();

    try wekua.tensor.random.uniform(T, cmd, tensor, null, null, null);

    const buffer = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(buffer);

    try wekua.tensor.memory.writeToBuffer(T, tensor, cmd, buffer);

    const sigmoid_layer = wekua.nn.activation.Sigmoid(T).init();
    try sigmoid_layer.run(cmd, tensor);

    const actual = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(actual);

    try wekua.tensor.memory.writeToBuffer(T, tensor, cmd, actual);

    if (is_complex) {
        for (0..(tensor.dimensions.number_of_elements_without_padding / 2)) |i| {
            const index = i * 2;
            const real_value = buffer[index];
            const imag_value = buffer[index + 1];

            const denominator = 1 + std.math.exp(-real_value);
            const output = (1.0 / denominator) * std.math.cos(imag_value);
            const ioutput = (-1.0 / denominator) * std.math.sin(imag_value);

            try std.testing.expectApproxEqAbs(output, actual[index], 0.001);
            try std.testing.expectApproxEqAbs(ioutput, actual[index + 1], 0.001);
        }
    }else{
        for (buffer, actual) |v, a| {
            const output = 1.0 / (1.0 + std.math.exp(-v));
            try std.testing.expectApproxEqAbs(output, a, 0.001);
        }
    }
}

test "Sigmoid" {
    const ctx = try wekua.core.Context.init_from_device_type(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.all,
    );
    defer ctx.release();

    const values: [2]bool = .{ false, true };

    inline for (&.{f32, f64}) |T| {
        for (&values) |is_complex| {
            for (&values) |vectors_enabled| {
                try test_sigmoid(T, std.testing.allocator, ctx, is_complex, vectors_enabled);
            }
        }
    }
}
