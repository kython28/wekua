const std = @import("std");
const wekua = @import("wekua");
const cl = wekua.opencl;

const tensor_module = wekua.tensor_module;
const nn = wekua.nn;

// This program demonstrates a simple neural network implementation
// using the Wekua library for machine learning in Zig.
// It creates a binary classification model to learn the XOR logic gate.

const FloatTensor = wekua.Tensor(f32);
const FloatLinear = nn.layer_module.linear_module.Linear(f32);
const FloatOptimizer = nn.optimizer_module.Optimizer(f32);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Create an OpenCL context using the best available device
    // This allows for potential GPU or specialized hardware acceleration
    const context = try wekua.core.Context.initFromDeviceType(
        gpa.allocator(), // Memory allocator
        null, // No specific platform constraints
        cl.device.Type.all, // Use any available device type
    );
    defer context.deinit(); // Ensure context is properly cleaned up

    // Select the first available command queue from the context
    const command_queue = &context.command_queues[0];

    // Create a pipeline for managing OpenCL command execution and dependencies
    const pipeline = try wekua.core.Pipeline.init(command_queue);
    defer pipeline.deinit();

    // XOR truth table inputs and expected outputs
    // Inputs: [1,1], [0,1], [1,0], [0,0]
    // Outputs: 0, 1, 1, 0
    const expected_outputs_buf: []const f32 = &.{ 0, 1, 1, 0 };

    const inputs_buf: []const f32 = &.{
        1, 1, // Case 1: 1 XOR 1 = 0
        0, 1, // Case 2: 0 XOR 1 = 1
        1, 0, // Case 3: 1 XOR 0 = 1
        0, 0, // Case 4: 0 XOR 0 = 0
    };

    // Create tensors on the OpenCL device (potentially a GPU)
    // inputs: 4 samples, 2 features per sample (XOR input)
    // expected_outputs: 4 samples, 1 output per sample
    const inputs = try FloatTensor.alloc(context, pipeline, &.{ 4, 2 }, .{});
    defer inputs.release(pipeline);

    const expected_outputs = try FloatTensor.alloc(context, pipeline, &.{ 4, 1 }, .{});
    defer expected_outputs.release(pipeline);

    // Transfer host (CPU) memory to device memory
    try tensor_module.memory.readFromBuffer(f32, pipeline, inputs, inputs_buf);
    try tensor_module.memory.readFromBuffer(f32, pipeline, expected_outputs, expected_outputs_buf);

    // Create a sequential neural network model
    // Architecture:
    // - Input layer: 2 neurons (XOR input)
    // - Hidden layer: 10 neurons with sigmoid activation
    // - Output layer: 1 neuron with sigmoid activation (binary classification)
    const seq_layers = try nn.layer_module.sequential_module.Sequential(f32).init(context.allocator);
    defer seq_layers.deinit(pipeline);

    const activation_layer = nn.activation_module.Sigmoid(f32).init();

    {
        // First layer: 2 input neurons to 10 hidden neurons
        const layer1 = try FloatLinear.init(
            context,
            pipeline,
            2, // Input dimensions
            10, // Output dimensions (hidden neurons)
            activation_layer,
            .{},
        );
        errdefer layer1.deinit(pipeline);

        try seq_layers.append(layer1);
    }

    {
        // Second layer: 10 hidden neurons to 1 output neuron
        const layer2 = try FloatLinear.init(
            context,
            pipeline,
            10, // Input dimensions (from previous layer)
            1, // Output dimensions (binary classification)
            activation_layer,
            .{},
        );
        errdefer layer2.deinit(pipeline);

        try seq_layers.append(layer2);
    }

    const layers = seq_layers.layer();

    // Initialize a cache for neural network training
    const cache = try nn.layer_module.Cache(f32).init(context, pipeline, 4, &.{&layers});
    defer cache.deinit(pipeline);

    const optimizer = try FloatOptimizer.GD.init(context.allocator, .{ .lr = 1 });
    defer optimizer.deinit(pipeline);

    const layer_cache = cache.getLayerCache(0);

    // Training loop
    // - Perform 300 iterations of forward and backward propagation
    // - Uses Mean Squared Error (MSE) as the loss function
    // - Gradient Descent (GD) optimizer with learning rate of 1
    for (0..300) |_| {
        // Forward pass: compute model predictions
        const output = try layers.forward(pipeline, inputs, layer_cache);

        // Compute loss using Mean Squared Error
        try nn.loss_module.mse(f32, true, pipeline, output, expected_outputs, &cache, null);

        // Backward pass: compute gradients
        try layers.backward(pipeline, layer_cache, inputs, null);

        // Update model weights using Gradient Descent
        try optimizer.step(pipeline, &cache);
    }

    // Final prediction after training
    const output = try layers.forward(pipeline, inputs, layer_cache);
    pipeline.waitAndCleanup();

    // Display the trained model's predictions
    try tensor_module.print(f32, pipeline, output);

    // Display the original expected outputs for comparison
    try tensor_module.print(f32, pipeline, expected_outputs);
}
