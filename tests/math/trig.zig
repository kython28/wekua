const wekua = @import("wekua");
const cl = wekua.opencl;
const std = @import("std");

const functions_names = [_][]const u8{
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
};

fn test_trig_function(
    comptime T: type,
    comptime function_name: []const u8,
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

    const buffer2 = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(buffer2);

    try wekua.tensor.memory.writeToBuffer(T, tensor, command_queue, buffer1);

    const wekua_trig_function = @field(wekua.math.trig, function_name);
    const std_trig_function = @field(std.math, function_name);

    try wekua_trig_function(T, command_queue, tensor);
    try wekua.tensor.memory.writeToBuffer(T, tensor, command_queue, buffer2);

    const eps = 1e-3;
    if (is_complex) {
        // TODO: Implement complex version
    }else{

        for (buffer1, buffer2) |e, a| {
            try std.testing.expectApproxEqAbs(std_trig_function(e), a, eps);
        }
    }
}

test "Trigonometric functions" {
    const ctx = try wekua.core.Context.create_from_best_device(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.all,
    );
    defer ctx.release();

    const command_queue = &ctx.command_queues[0];

    inline for (functions_names) |function_name| {
        inline for (&.{f32, f64}) |T| {
            inline for (&.{true, false}) |is_complex| {
                inline for (&.{true, false}) |vector_enabled| {
                    try test_trig_function(T, function_name, ctx, command_queue, is_complex, vector_enabled);
                }
            }
        }
    }
}
