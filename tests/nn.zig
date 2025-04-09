const wekua = @import("wekua");
const cl = wekua.opencl;
const std = @import("std");

fn solve_and(
    comptime T: type,
    context: *wekua.core.Context,
) !void {
    const command_queue = &context.command_queues[0];
    if (!command_queue.typeIsSupported(T)) return;

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


    try wekua.tensor.memory.readFromBuffer(T, inputs, command_queue, inputs_buf);
    try wekua.tensor.memory.readFromBuffer(T, expected_outputs, command_queue, expected_outputs_buf);

    const activation_layer = wekua.nn.activation.Sigmoid(T).init();

    const Linear = wekua.nn.layer.Linear(T);
    const Optimizer = wekua.nn.optimizer.Optimizer(T);

    const layer1 = try Linear.init(
        context,
        2,
        1,
        activation_layer,
        .{}
    );
    defer layer1.deinit();

    const cache = try wekua.nn.layer.Cache(T).init(context, 4, &.{&layer1});
    defer cache.deinit();

    const optimizer = try Optimizer.GD.init(context.allocator, 1, null);
    defer optimizer.deinit();

    var prediction_buffer: [4]T = undefined;

    const layer_cache = cache.getLayerCache(0);
    for (0..300) |_| {
        const output = try layer1.forward(command_queue, inputs, layer_cache);
        try wekua.nn.loss.mse(T, true, command_queue, output, expected_outputs, &cache, null, null);

        try layer1.backward(command_queue, layer_cache, inputs, null);
        try optimizer.step(command_queue, &cache);
    }

    const output = try layer1.forward(command_queue, inputs, cache.slots[0].cache);

    try wekua.tensor.memory.writeToBuffer(T, output, command_queue, &prediction_buffer);
    for (prediction_buffer, expected_outputs_buf) |p, e| {
        try std.testing.expectApproxEqAbs(e, p, 0.2);
    }
}

test "Solving `AND` problem" {
    const ctx = try wekua.core.Context.create_from_best_device(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.all,
    );
    defer ctx.release();

    try solve_and(f32, ctx);
    try solve_and(f64, ctx);
}


fn solve_or(
    comptime T: type,
    context: *wekua.core.Context,
) !void {
    const command_queue = &context.command_queues[0];
    if (!command_queue.typeIsSupported(T)) return;

    const expected_outputs_buf: []const T = &.{ 1, 1, 1, 0 };

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


    try wekua.tensor.memory.readFromBuffer(T, inputs, command_queue, inputs_buf);
    try wekua.tensor.memory.readFromBuffer(T, expected_outputs, command_queue, expected_outputs_buf);

    const activation_layer = wekua.nn.activation.Sigmoid(T).init();

    const Linear = wekua.nn.layer.Linear(T);
    const Optimizer = wekua.nn.optimizer.Optimizer(T);

    const layer1 = try Linear.init(
        context,
        2,
        1,
        activation_layer,
        .{}
    );
    defer layer1.deinit();

    const cache = try wekua.nn.layer.Cache(T).init(context, 4, &.{&layer1});
    defer cache.deinit();

    const optimizer = try Optimizer.GD.init(context.allocator, 1, null);
    defer optimizer.deinit();

    var prediction_buffer: [4]T = undefined;

    const layer_cache = cache.getLayerCache(0);
    for (0..300) |_| {
        const output = try layer1.forward(command_queue, inputs, layer_cache);
        try wekua.nn.loss.mse(T, true, command_queue, output, expected_outputs, &cache, null, null);

        try layer1.backward(command_queue, layer_cache, inputs, null);
        try optimizer.step(command_queue, &cache);
    }

    const output = try layer1.forward(command_queue, inputs, cache.slots[0].cache);

    try wekua.tensor.memory.writeToBuffer(T, output, command_queue, &prediction_buffer);
    for (prediction_buffer, expected_outputs_buf) |p, e| {
        try std.testing.expectApproxEqAbs(e, p, 0.2);
    }
}

test "Solving `OR` problem" {
    const ctx = try wekua.core.Context.create_from_best_device(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.all,
    );
    defer ctx.release();

    try solve_or(f32, ctx);
    try solve_or(f64, ctx);
}

fn solve_xor(
    comptime T: type,
    context: *wekua.core.Context,
) !void {
    const command_queue = &context.command_queues[0];
    if (!command_queue.typeIsSupported(T)) return;

    const expected_outputs_buf: []const T = &.{ 0, 1, 1, 0 };

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

    try wekua.tensor.memory.readFromBuffer(T, inputs, command_queue, inputs_buf);
    try wekua.tensor.memory.readFromBuffer(T, expected_outputs, command_queue, expected_outputs_buf);

    const activation_layer = wekua.nn.activation.Sigmoid(T).init();

    const Linear = wekua.nn.layer.Linear(T);
    const Optimizer = wekua.nn.optimizer.Optimizer(T);

    const seq_layers = try wekua.nn.layer.Sequential(T).init(context.allocator);
    defer seq_layers.deinit();

    try seq_layers.append(
        try Linear.init(
            context,
            2,
            10,
            activation_layer,
            .{}
        ),
    );
    try seq_layers.append(
        try Linear.init(
            context,
            10,
            1,
            activation_layer,
            .{}
        ),
    );

    const layers = seq_layers.layer();

    const cache = try wekua.nn.layer.Cache(T).init(context, 4, &.{&layers});
    defer cache.deinit();

    const optimizer = try Optimizer.GD.init(context.allocator, 1, null);
    defer optimizer.deinit();

    var prediction_buffer: [4]T = undefined;

    const layer_cache = cache.getLayerCache(0);
    for (0..300) |_| {
        const output = try layers.forward(command_queue, inputs, layer_cache);
        try wekua.nn.loss.mse(T, true, command_queue, output, expected_outputs, &cache, null, null);

        try layers.backward(command_queue, layer_cache, inputs, null);
        try optimizer.step(command_queue, &cache);
    }

    const output = try layers.forward(command_queue, inputs, cache.slots[0].cache);

    try wekua.tensor.memory.writeToBuffer(T, output, command_queue, &prediction_buffer);
    for (prediction_buffer, expected_outputs_buf) |p, e| {
        try std.testing.expectApproxEqAbs(e, p, 0.2);
    }
}

test "Solving `XOR` problem" {
    const ctx = try wekua.core.Context.create_from_best_device(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.all,
    );
    defer ctx.release();

    try solve_xor(f32, ctx);
    try solve_xor(f64, ctx);
}
