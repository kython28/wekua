const std = @import("std");
const wekua = @import("wekua");
const cl = wekua.opencl;

// This program demonstrates a simple neural network implementation 
// using the Wekua library for machine learning in Zig.
// It creates a binary classification model to learn the XOR logic gate.

const FloatTensor = wekua.Tensor(f32);
const FloatLinear = wekua.nn.layer.Linear(f32);
const FloatOptimizer = wekua.nn.optimizer.Optimizer(f32);

pub fn main() !void {
    // Create an OpenCL context using the best available device
    // This allows for potential GPU or specialized hardware acceleration
    const context = try wekua.core.Context.create_from_best_device(
        std.heap.c_allocator,  // Memory allocator
        null,                  // No specific platform constraints
        cl.device.enums.device_type.all,  // Use any available device type
    );
    defer context.release();  // Ensure context is properly cleaned up

    // Select the first available command queue from the context
    // In OpenCL, a command queue is associated with a specific device
    // and is used to submit commands for execution on that device.
    // Here, we're choosing the first command queue, which typically 
    // corresponds to the best or primary device selected during context creation.
    const command_queue = &context.command_queues[0];

    // XOR truth table inputs and expected outputs
    // Represents the classic XOR logic gate problem
    // Inputs: [1,1], [0,1], [1,0], [0,0]
    // Outputs: 0, 1, 1, 0
    const expected_outputs_buf: []const f32 = &.{ 0, 1, 1, 0 };

    const inputs_buf: []const f32 = &.{
        1, 1,  // Case 1: 1 XOR 1 = 0
        0, 1,  // Case 2: 0 XOR 1 = 1
        1, 0,  // Case 3: 1 XOR 0 = 1
        0, 0,  // Case 4: 0 XOR 0 = 0
    };
    
    // Create tensors on the OpenCL device (potentially a GPU)
    // Allocate memory directly on the device with specific dimensions
    // inputs: 4 samples, 2 features per sample (XOR input)
    // expected_outputs: 4 samples, 1 output per sample
    const inputs = try FloatTensor.alloc(context, &.{ 4, 2 }, .{});
    defer inputs.release();  // Ensure device memory is freed after use

    const expected_outputs = try FloatTensor.alloc(context, &.{ 4, 1 }, .{});
    defer expected_outputs.release();  // Ensure device memory is freed after use

    // Transfer host (CPU) memory to device memory
    // This copies the input data from local buffers to the device-allocated tensors
    // Using the command queue ensures efficient memory transfer across the device
    // readFromBuffer handles the cross-device memory transfer automatically
    try wekua.tensor.memory.readFromBuffer(f32, inputs, command_queue, inputs_buf);
    try wekua.tensor.memory.readFromBuffer(f32, expected_outputs, command_queue, expected_outputs_buf);

    // Create a sequential neural network model
    // Architecture: 
    // - Input layer: 2 neurons (XOR input)
    // - Hidden layer: 10 neurons with sigmoid activation
    // - Output layer: 1 neuron with sigmoid activation (binary classification)
    const seq_layers = try wekua.nn.layer.Sequential(f32).init(context.allocator);
    defer seq_layers.deinit();

    const activation_layer = wekua.nn.activation.Sigmoid(f32).init();

    {
        // First layer: 2 input neurons to 10 hidden neurons
        const layer1 = try FloatLinear.init(
            context,
            2,      // Input dimensions
            10,     // Output dimensions (hidden neurons)
            activation_layer,
            .{}
        );
        errdefer layer1.deinit();

        // Add layers to the sequential model
        try seq_layers.append(layer1);
    }

    {
        // Second layer: 10 hidden neurons to 1 output neuron
        const layer2 = try FloatLinear.init(
            context,
            10,     // Input dimensions (from previous layer)
            1,      // Output dimensions (binary classification)
            activation_layer,
            .{}
        );
        errdefer layer2.deinit();

        // Add layers to the sequential model
        try seq_layers.append(layer2);
    }

    const layers = seq_layers.layer();

    // Initialize a cache for neural network training
    // Purpose of the cache:
    // 1. Allocate memory for intermediate layer computations
    // 2. Store forward pass outputs for each layer
    // 3. Preserve gradients and derivatives during backpropagation
    // 4. Optimize memory usage during training
    // 
    // Parameters:
    // - context: OpenCL context for memory allocation
    // - 4: Number of training samples
    // - &.{&layers}: Layers to prepare cache for
    const cache = try wekua.nn.layer.Cache(f32).init(context, 4, &.{&layers});
    defer cache.deinit();  // Ensure cache memory is properly freed

    const optimizer = try FloatOptimizer.GD.init(context.allocator, 1, null);
    defer optimizer.deinit();

    const layer_cache = cache.getLayerCache(0);
    // Training loop
    // - Perform 300 iterations of forward and backward propagation
    // - Uses Mean Squared Error (MSE) as the loss function
    // - Gradient Descent (GD) optimizer with learning rate of 1
    for (0..300) |_| {
        // Forward pass: compute model predictions
        const output = try layers.forward(command_queue, inputs, layer_cache);
        
        // Compute loss using Mean Squared Error
        try wekua.nn.loss.mse(f32, true, command_queue, output, expected_outputs, &cache, null, null);

        // Backward pass: compute gradients
        try layers.backward(command_queue, layer_cache, inputs, null);
        
        // Update model weights using Gradient Descent
        try optimizer.step(command_queue, &cache);
    }

    // Final prediction after training
    // Print the model's predictions and the expected outputs
    const output = try layers.forward(command_queue, inputs, layer_cache);

    // Display the trained model's predictions
    try wekua.tensor.print(f32, command_queue, output);
    
    // Display the original expected outputs for comparison
    try wekua.tensor.print(f32, command_queue, expected_outputs);
}
