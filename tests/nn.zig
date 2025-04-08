const wekua = @import("wekua");
const cl = wekua.opencl;
const std = @import("std");

fn solve_and(
    comptime T: type,
    context: *wekua.core.Context,
    enable_bias: bool,
) !void {
    const expected_outputs_buf: []const T = &.{ 1, 0, 0, 0 };

    const inputs_buf: []const T = &.{
        1, 1,
        0, 1,
        1, 0,
        0, 0,
    };

    const Tensor = wekua.Tensor(T);

    const inputs = try Tensor.alloc(context, &.{ 4, 2 }, .{});
    defer inputs.release();

    const expected_outputs = try Tensor.alloc(context, &.{ 4, 1 }, .{});
    defer expected_outputs.release();

    const command_queue = &context.command_queues[0];

    try wekua.tensor.memory.readFromBuffer(T, inputs, command_queue, inputs_buf);
    try wekua.tensor.memory.readFromBuffer(T, expected_outputs, command_queue, expected_outputs_buf);

    const activation_layer = wekua.nn.activation.Sigmoid(T).init();

    const Linear = wekua.nn.layer.Linear(T);
    const Optimizer = wekua.nn.optimizer.Optimizer(T);

    const layer1 = try Linear.init(
        context,
        2,
        1,
        1,
        enable_bias,
        activation_layer,
        false,
    );
    defer layer1.deinit();

    const cache = try Linear.Cache.init(context, 4, &.{&layer1});
    defer cache.deinit();

    const optimizer = try Optimizer.GD.init(context.allocator, 0.001, null);
    defer optimizer.deinit();

    for (0..100) |_| {
        const output = try layer1.forward(command_queue, inputs, cache.slots[0].cache);
        try wekua.nn.loss.mse(T, true, command_queue, output, expected_outputs, &cache, null, null);

        try layer1.backward(command_queue, cache.slots[0].cache, inputs, null);
        try optimizer.step(command_queue, &cache);
    }

    const output = try layer1.forward(command_queue, inputs, cache.slots[0].cache);

    var prediction_buffer: [4]T = undefined;

    try wekua.tensor.memory.writeToBuffer(T, output, command_queue, &prediction_buffer);
    std.log.warn("Prediction: {any}", .{&prediction_buffer});

    for (prediction_buffer, expected_outputs_buf) |p, e| {
        try std.testing.expectApproxEqAbs(e, p, 0.1);
    }
}

test "Solving `AND` problem" {
    const ctx = try wekua.core.Context.create_from_best_device(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.gpu,
    );
    defer ctx.release();

    try solve_and(f32, ctx, true);
    try solve_and(f64, ctx, true);

    try solve_and(f32, ctx, false);
    try solve_and(f64, ctx, false);
}
