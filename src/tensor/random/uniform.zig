const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const CommandQueue = core.CommandQueue;
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const helpers = @import("../helpers.zig");

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const uniform_random_cl_kernel: []const u8 = @embedFile("kernels/uniform.cl");

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    range_defined: bool,
) TensorErrors!cl.kernel.Kernel {
    const kernels_set = try KernelsSet.getKernelSet(
        command_queue,
        .RandomUniform,
        core.types.SupportedTypes.len * 2,
    );
    const index: usize = @as(usize, core.types.getTypeIndex(T)) * 2 + @intFromBool(range_defined);
    if (kernels_set.kernels.?[index]) |v| return v;

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    const allocator = command_queue.context.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(
        allocator,
        "-DRANGE_DEFINED={d}",
        .{@intFromBool(range_defined)},
    );
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .vectors_enabled = false,
            .kernel_name = "uniform",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        uniform_random_cl_kernel,
    );

    kernels_set.kernels.?[index] = kernel;
    kernels_set.programs.?[index] = program;

    return kernel;
}

pub fn uniform(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    seed: ?u64,
    min_value: ?core.types.getType(T),
    max_value: ?core.types.getType(T),
) TensorErrors!void {
    const range_defined = min_value != null or max_value != null;
    const command_queue = pipeline.command_queue;

    const kernel = try getKernel(
        T,
        command_queue,
        range_defined,
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    const global_seed = seed orelse @as(u64, @bitCast(std.time.timestamp()));

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try setArg(kernel, 1, @sizeOf(u64), @ptrCast(&tensor.memory_layout.row_pitch));
    try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.memory_layout.slice_pitch));
    try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&global_seed));

    if (range_defined) {
        const SubType = core.types.getType(T);
        const min = min_value orelse switch (@typeInfo(SubType)) {
            .int => std.math.minInt(SubType),
            .float => -std.math.floatMax(SubType),
            else => @compileError("Unsupported type"),
        };

        const max = max_value orelse switch (@typeInfo(SubType)) {
            .int => std.math.maxInt(SubType),
            .float => std.math.floatMax(SubType),
            else => @compileError("Unsupported type"),
        };

        try setArg(kernel, 4, @sizeOf(SubType), @ptrCast(&min));
        try setArg(kernel, 5, @sizeOf(SubType), @ptrCast(&max));
    }

    // TODO: Adapt code to use views
    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        &tensor.work_configuration.global_work_items_without_vectors,
        &tensor.work_configuration.local_work_items_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

// -----------------------------------------------------------------------------
// Unit Tests
const testing = std.testing;

const memory = @import("../memory/main.zig");

const StatResult = struct {
    real_mean: f64,
    imag_mean: f64,
    real_stddev: f64,
    imag_stddev: f64,
};

fn calculateStatistics(comptime T: type, data: []const T) StatResult {
    const SubType = core.types.getType(T);
    var real_sum: f64 = 0.0;
    var imag_sum: f64 = 0.0;

    // Calculate means
    if (comptime core.types.isComplex(T)) {
        for (data) |val| {
            const real_val: f64 = switch (@typeInfo(SubType)) {
                .int => @floatFromInt(val.real),
                .float => @floatCast(val.real),
                else => unreachable,
            };
            const imag_val: f64 = switch (@typeInfo(SubType)) {
                .int => @floatFromInt(val.imag),
                .float => @floatCast(val.imag),
                else => unreachable,
            };
            real_sum += real_val;
            imag_sum += imag_val;
        }
    } else {
        for (data) |val| {
            const float_val: f64 = switch (@typeInfo(SubType)) {
                .int => @floatFromInt(val),
                .float => @floatCast(val),
                else => unreachable,
            };
            real_sum += float_val;
        }
    }

    const n: f64 = @floatFromInt(data.len);
    const real_mean = real_sum / n;
    const imag_mean = imag_sum / n;

    // Calculate standard deviations
    var real_variance: f64 = 0.0;
    var imag_variance: f64 = 0.0;

    if (comptime core.types.isComplex(T)) {
        for (data) |val| {
            const real_val: f64 = switch (@typeInfo(SubType)) {
                .int => @floatFromInt(val.real),
                .float => @floatCast(val.real),
                else => unreachable,
            };
            const imag_val: f64 = switch (@typeInfo(SubType)) {
                .int => @floatFromInt(val.imag),
                .float => @floatCast(val.imag),
                else => unreachable,
            };

            const real_diff = real_val - real_mean;
            const imag_diff = imag_val - imag_mean;
            real_variance += real_diff * real_diff;
            imag_variance += imag_diff * imag_diff;
        }
    } else {
        for (data) |val| {
            const float_val: f64 = switch (@typeInfo(SubType)) {
                .int => @floatFromInt(val),
                .float => @floatCast(val),
                else => unreachable,
            };
            const diff = float_val - real_mean;
            real_variance += diff * diff;
        }
    }

    return .{
        .real_mean = real_mean,
        .imag_mean = imag_mean,
        .real_stddev = @sqrt(real_variance / n),
        .imag_stddev = @sqrt(imag_variance / n),
    };
}

test "uniform - 1D tensor without range for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{100};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Generate uniform random values
            try uniform(T, pipeline, tensor, 42, null, null);

            // Read back and verify
            const buffer_size = shape[0];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // For floating point types, verify values are in [0, 1]
            const SubType = core.types.getType(T);
            if (@typeInfo(SubType) == .float) {
                for (output_buffer) |val| {
                    if (comptime core.types.isComplex(T)) {
                        try testing.expect(val.real >= 0.0 and val.real <= 1.0);
                        try testing.expect(val.imag >= 0.0 and val.imag <= 1.0);
                    } else {
                        try testing.expect(val >= 0.0 and val <= 1.0);
                    }
                }
            }
        }
    }
}

test "uniform - 2D tensor without range for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 10, 10 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Generate uniform random values
            try uniform(T, pipeline, tensor, 123, null, null);

            // Read back and verify
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // For floating point types, verify values are in [0, 1]
            const SubType = core.types.getType(T);
            if (@typeInfo(SubType) == .float) {
                for (output_buffer) |val| {
                    if (comptime core.types.isComplex(T)) {
                        try testing.expect(val.real >= 0.0 and val.real <= 1.0);
                        try testing.expect(val.imag >= 0.0 and val.imag <= 1.0);
                    } else {
                        try testing.expect(val >= 0.0 and val <= 1.0);
                    }
                }
            }
        }
    }
}

test "uniform - 3D tensor without range for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 5, 4, 5 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Generate uniform random values
            try uniform(T, pipeline, tensor, 999, null, null);

            // Read back and verify
            const buffer_size = shape[0] * shape[1] * shape[2];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // For floating point types, verify values are in [0, 1]
            const SubType = core.types.getType(T);
            if (@typeInfo(SubType) == .float) {
                for (output_buffer) |val| {
                    if (comptime core.types.isComplex(T)) {
                        try testing.expect(val.real >= 0.0 and val.real <= 1.0);
                        try testing.expect(val.imag >= 0.0 and val.imag <= 1.0);
                    } else {
                        try testing.expect(val >= 0.0 and val <= 1.0);
                    }
                }
            }
        }
    }
}

test "uniform - with custom range for integer types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{100};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const SubType = core.types.getType(T);
            if (@typeInfo(SubType) == .int) {
                const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer tensor.release(pipeline);

                const min_val: SubType = 10;
                const max_val: SubType = 50;

                // Generate uniform random values with range
                try uniform(T, pipeline, tensor, 42, min_val, max_val);

                // Read back and verify
                const buffer_size = shape[0];
                const output_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(output_buffer);

                try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
                pipeline.waitAndCleanup();

                // Verify all values are in range
                for (output_buffer) |val| {
                    if (comptime core.types.isComplex(T)) {
                        try testing.expect(val.real >= 10 and val.real <= 50);
                        try testing.expect(val.imag >= 10 and val.imag <= 50);
                    } else {
                        try testing.expect(val >= 10 and val <= 50);
                    }
                }
            }
        }
    }
}

test "uniform - with custom range for floating point types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{100};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const SubType = core.types.getType(T);
            if (@typeInfo(SubType) == .float) {
                const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer tensor.release(pipeline);

                const min_val: SubType = -5.0;
                const max_val: SubType = 5.0;

                // Generate uniform random values with range
                try uniform(T, pipeline, tensor, 42, min_val, max_val);

                // Read back and verify
                const buffer_size = shape[0];
                const output_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(output_buffer);

                try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
                pipeline.waitAndCleanup();

                // Verify all values are in range
                for (output_buffer) |val| {
                    if (comptime core.types.isComplex(T)) {
                        try testing.expect(val.real >= -5.0 and val.real <= 5.0);
                        try testing.expect(val.imag >= -5.0 and val.imag <= 5.0);
                    } else {
                        try testing.expect(val >= -5.0 and val <= 5.0);
                    }
                }
            }
        }
    }
}

test "uniform - different seeds produce different values" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{50};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor1 = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor1.release(pipeline);

            const tensor2 = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor2.release(pipeline);

            // Generate with different seeds
            try uniform(T, pipeline, tensor1, 42, null, null);
            try uniform(T, pipeline, tensor2, 1337, null, null);

            // Read back both
            const buffer_size = shape[0];
            const output_buffer1 = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer1);

            const output_buffer2 = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer2);

            try memory.writeToBuffer(T, pipeline, tensor1, output_buffer1);
            try memory.writeToBuffer(T, pipeline, tensor2, output_buffer2);
            pipeline.waitAndCleanup();

            // Verify at least some values are different
            var differences: usize = 0;
            for (output_buffer1, output_buffer2) |val1, val2| {
                if (comptime core.types.isComplex(T)) {
                    if (val1.real != val2.real or val1.imag != val2.imag) {
                        differences += 1;
                    }
                } else {
                    if (val1 != val2) {
                        differences += 1;
                    }
                }
            }

            // At least 80% of values should be different with different seeds
            try testing.expect(differences > buffer_size * 4 / 5);
        }
    }
}

test "uniform - same seed produces same values" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{50};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor1 = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor1.release(pipeline);

            const tensor2 = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor2.release(pipeline);

            // Generate with same seed
            try uniform(T, pipeline, tensor1, 42, null, null);
            try uniform(T, pipeline, tensor2, 42, null, null);

            // Read back both
            const buffer_size = shape[0];
            const output_buffer1 = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer1);

            const output_buffer2 = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer2);

            try memory.writeToBuffer(T, pipeline, tensor1, output_buffer1);
            try memory.writeToBuffer(T, pipeline, tensor2, output_buffer2);
            pipeline.waitAndCleanup();

            // Verify all values are identical
            for (output_buffer1, output_buffer2) |val1, val2| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(val1.real, val2.real);
                    try testing.expectEqual(val1.imag, val2.imag);
                } else {
                    try testing.expectEqual(val1, val2);
                }
            }
        }
    }
}

test "uniform - statistical properties for floating point without range" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10000};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const SubType = core.types.getType(T);
            if (@typeInfo(SubType) == .float) {
                const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer tensor.release(pipeline);

                // Generate uniform random values
                try uniform(T, pipeline, tensor, 42, null, null);

                // Read back
                const buffer_size = shape[0];
                const output_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(output_buffer);

                try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
                pipeline.waitAndCleanup();

                // Calculate statistics
                const stats = calculateStatistics(T, output_buffer);

                // For uniform distribution [0, 1]:
                // Expected mean = 0.5
                // Expected stddev = sqrt(1/12) ≈ 0.289
                const expected_mean = 0.5;
                const expected_stddev = 0.289;

                // Allow 5% tolerance for mean and stddev
                const mean_tolerance = 0.05;
                const stddev_tolerance = 0.05;

                // Verify real part
                try testing.expect(@abs(stats.real_mean - expected_mean) < mean_tolerance);
                try testing.expect(@abs(stats.real_stddev - expected_stddev) < stddev_tolerance);

                // For complex types, verify imaginary part too
                if (comptime core.types.isComplex(T)) {
                    try testing.expect(@abs(stats.imag_mean - expected_mean) < mean_tolerance);
                    try testing.expect(@abs(stats.imag_stddev - expected_stddev) < stddev_tolerance);
                }
            }
        }
    }
}

test "uniform - statistical properties for integer with range" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10000};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const SubType = core.types.getType(T);
            if (@typeInfo(SubType) == .int) {
                const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer tensor.release(pipeline);

                const min_val: SubType = 0;
                const max_val: SubType = 100;

                // Generate uniform random values with range
                try uniform(T, pipeline, tensor, 42, min_val, max_val);

                // Read back
                const buffer_size = shape[0];
                const output_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(output_buffer);

                try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
                pipeline.waitAndCleanup();

                // Calculate statistics
                const stats = calculateStatistics(T, output_buffer);

                // Expected mean for uniform [0, 100] ≈ 50
                const expected_mean = 50.0;
                const mean_tolerance = 5.0;

                // Verify real part
                try testing.expect(@abs(stats.real_mean - expected_mean) < mean_tolerance);

                // For complex types, verify imaginary part (same range [0, 100] -> mean ≈ 50)
                if (comptime core.types.isComplex(T)) {
                    try testing.expect(@abs(stats.imag_mean - expected_mean) < mean_tolerance);
                }
            }
        }
    }
}

test "uniform - complex types with custom range" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5000};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            if (comptime core.types.isComplex(T)) {
                const SubType = core.types.getType(T);
                const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer tensor.release(pipeline);

                // Same range applies to both real and imaginary parts
                const min_val: SubType = if (@typeInfo(SubType) == .int)
                    50
                else
                    10.0;

                const max_val: SubType = if (@typeInfo(SubType) == .int)
                    100
                else
                    100.0;

                // Generate uniform random values with range
                try uniform(T, pipeline, tensor, 12345, min_val, max_val);

                // Read back
                const buffer_size = shape[0];
                const output_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(output_buffer);

                try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
                pipeline.waitAndCleanup();

                // Verify all values are in range
                for (output_buffer) |val| {
                    if (@typeInfo(SubType) == .int) {
                        try testing.expect(val.real >= 50 and val.real <= 100);
                        try testing.expect(val.imag >= 50 and val.imag <= 100);
                    } else {
                        try testing.expect(val.real >= 10.0 and val.real <= 100.0);
                        try testing.expect(val.imag >= 10.0 and val.imag <= 100.0);
                    }
                }

                // Calculate and verify statistics
                // Both real and imag should have mean ≈ 0 (symmetric range around 0)
                const stats = calculateStatistics(T, output_buffer);

                if (@typeInfo(SubType) == .int) {
                    // Range [-50, 50] -> mean ≈ 0 for both parts
                    try testing.expect(@abs(stats.real_mean - 0.0) < 5.0);
                    try testing.expect(@abs(stats.imag_mean - 0.0) < 5.0);
                } else {
                    // Range [-10.0, 10.0] -> mean ≈ 0.0 for both parts
                    try testing.expect(@abs(stats.real_mean - 0.0) < 1.0);
                    try testing.expect(@abs(stats.imag_mean - 0.0) < 1.0);
                }
            }
        }
    }
}
