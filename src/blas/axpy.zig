const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const axpy_Kernel: []const u8 = @embedFile("kernels/axpy.cl");

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    comptime kernel_name: []const u8,
    vectors_enabled: bool,
    has_alpha: bool,
    substract: bool,
) TensorErrors!cl.kernel.Kernel {
    const SUPPORTED_TYPES = core.types.SUPPORTED_TYPES;
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .AXPY, SUPPORTED_TYPES.len * 2 * 2 * 2);

    var kernel_index: usize = @intFromBool(vectors_enabled) * (2 * 2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromBool(has_alpha) * (2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromBool(substract) * SUPPORTED_TYPES.len;
    kernel_index += @as(usize, core.types.getTypeIndex(T));

    if (kernels_set.kernels.?[kernel_index]) |v| return v;

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    const allocator = command_queue.context.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(
        allocator,
        "-DHAS_ALPHA={d} -DSUBSTRACT={d}",
        .{
            @intFromBool(has_alpha),
            @intFromBool(substract),
        },
    );
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .vectors_enabled = vectors_enabled,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        axpy_Kernel,
    );

    kernels_set.kernels.?[kernel_index] = kernel;
    kernels_set.programs.?[kernel_index] = program;

    return kernel;
}

inline fn isSubstracting(comptime T: type, alpha: T) bool {
    const SubType = core.types.getType(T);
    const is_complex = comptime core.types.isComplex(T);
    return switch (@typeInfo(SubType)) {
        .int => |int_info| blk: {
            if (int_info.signedness == .unsigned) {
                break :blk false;
            }

            if (is_complex) {
                break :blk (alpha.real == @as(SubType, -1) and (alpha.imag == 0 or alpha.imag == @as(SubType, -1)));
            }

            break :blk (alpha == @as(SubType, -1));
        },
        .float => blk: {
            const eps = comptime std.math.floatEps(SubType);
            if (is_complex) {
                break :blk (@abs(alpha.real + @as(SubType, 1)) < eps and (@abs(alpha.imag) < eps or @abs(alpha.imag + @as(SubType, 1)) < eps));
            }

            break :blk (@abs(alpha + @as(SubType, 1)) < eps);
        },
        else => @compileError("Type not supported"),
    };
}

pub fn axpy(
    comptime T: type,
    pipeline: *Pipeline,
    x: *Tensor(T),
    alpha: ?T,
    y: *Tensor(T),
) TensorErrors!void {
    try tensor_module.helpers.eqlTensorsShape(T, x, y);

    const command_queue = pipeline.command_queue;
    var has_alpha = false;
    const substract = blk: {
        if (alpha) |s| {
            const res = isSubstracting(T, s);
            has_alpha = !res;
            break :blk res;
        }
        break :blk false;
    };

    const vectors_enabled = x.flags.vectors_enabled and y.flags.vectors_enabled;
    const kernel = try getKernel(
        T,
        command_queue,
        "axpy",
        vectors_enabled,
        has_alpha,
        substract
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));

    const wekua_id = command_queue.wekua_id;
    var global_work_items: []const u64 = undefined;
    var local_work_items: []const u64 = undefined;
    var arg_index: u32 = 2;

    if (vectors_enabled) {
        global_work_items = @as([*]u64, @ptrCast(&x.memory_layout.number_of_vectors))[0..1];
        local_work_items = x.work_configuration.local_work_items_for_vectors_1d[wekua_id .. wekua_id + 1];
    } else {
        global_work_items = &x.work_configuration.global_work_items;
        local_work_items = &x.work_configuration.local_work_items[wekua_id];

        try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&x.memory_layout.slice_pitch_for_vectors));
        try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&x.memory_layout.row_pitch_for_vectors));
        try setArg(kernel, 4, @sizeOf(u64), @ptrCast(&y.memory_layout.slice_pitch_for_vectors));
        try setArg(kernel, 5, @sizeOf(u64), @ptrCast(&y.memory_layout.row_pitch_for_vectors));

        arg_index = 6;
    }

    if (has_alpha) {
        const _has_alpha = alpha.?;
        try setArg(kernel, arg_index, @sizeOf(T), @ptrCast(&_has_alpha));
    }

    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        global_work_items,
        local_work_items,
        prev_events,
        &new_event,
    );
    errdefer tensor_module.helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

// -----------------------------------------------------------------------------
// Unit Tests
const testing = std.testing;

const memory = tensor_module.memory;
const fill = tensor_module.fill;

test "axpy - basic operation y = alpha*x + y for 1D tensor" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const x = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const y = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer y.release(pipeline);

            const buffer_size = shape[0];

            // Initialize x with [1, 2, 3, 4, 5]
            const x_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(x_buffer);

            for (x_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i + 1), .imag = 0 },
                        .int => .{ .real = @intCast(i + 1), .imag = 0 },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i + 1),
                        .int => @intCast(i + 1),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, x, x_buffer);

            // Initialize y with [10, 20, 30, 40, 50]
            const y_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(y_buffer);

            for (y_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt((i + 1) * 10), .imag = 0 },
                        .int => .{ .real = @intCast((i + 1) * 10), .imag = 0 },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt((i + 1) * 10),
                        .int => @intCast((i + 1) * 10),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, y, y_buffer);

            // Perform y = 2*x + y
            const alpha: T = if (comptime core.types.isComplex(T))
                .{ .real = 2, .imag = 0 }
            else
                2;

            try axpy(T, pipeline, x, alpha, y);

            // Read result
            const result_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_buffer);

            try memory.writeToBuffer(T, pipeline, y, result_buffer);
            pipeline.waitAndCleanup();

            // Expected: [12, 24, 36, 48, 60] = 2*[1, 2, 3, 4, 5] + [10, 20, 30, 40, 50]
            for (result_buffer, 0..) |val, i| {
                const expected_val = (i + 1) * 2 + (i + 1) * 10;

                if (comptime core.types.isComplex(T)) {
                    const expected = switch (@typeInfo(core.types.getType(T))) {
                        .float => @as(core.types.getType(T), @floatFromInt(expected_val)),
                        .int => @as(core.types.getType(T), @intCast(expected_val)),
                        else => unreachable,
                    };
                    try testing.expectEqual(expected, val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    const expected = switch (@typeInfo(T)) {
                        .float => @as(T, @floatFromInt(expected_val)),
                        .int => @as(T, @intCast(expected_val)),
                        else => unreachable,
                    };
                    try testing.expectEqual(expected, val);
                }
            }
        }
    }
}

test "axpy - with alpha = null (direct sum) for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{4};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const x = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const y = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer y.release(pipeline);

            const buffer_size = shape[0];

            // Initialize x with [1, 2, 3, 4]
            const x_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(x_buffer);

            for (x_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i + 1), .imag = 0 },
                        .int => .{ .real = @intCast(i + 1), .imag = 0 },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i + 1),
                        .int => @intCast(i + 1),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, x, x_buffer);

            // Initialize y with [5, 6, 7, 8]
            const y_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(y_buffer);

            for (y_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i + 5), .imag = 0 },
                        .int => .{ .real = @intCast(i + 5), .imag = 0 },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i + 5),
                        .int => @intCast(i + 5),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, y, y_buffer);

            // Perform y = x + y (alpha = null)
            try axpy(T, pipeline, x, null, y);

            // Read result
            const result_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_buffer);

            try memory.writeToBuffer(T, pipeline, y, result_buffer);
            pipeline.waitAndCleanup();

            // Expected: [6, 8, 10, 12] = [1, 2, 3, 4] + [5, 6, 7, 8]
            for (result_buffer, 0..) |val, i| {
                const expected_val = (i + 1) + (i + 5);

                if (comptime core.types.isComplex(T)) {
                    const expected = switch (@typeInfo(core.types.getType(T))) {
                        .float => @as(core.types.getType(T), @floatFromInt(expected_val)),
                        .int => @as(core.types.getType(T), @intCast(expected_val)),
                        else => unreachable,
                    };
                    try testing.expectEqual(expected, val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    const expected = switch (@typeInfo(T)) {
                        .float => @as(T, @floatFromInt(expected_val)),
                        .int => @as(T, @intCast(expected_val)),
                        else => unreachable,
                    };
                    try testing.expectEqual(expected, val);
                }
            }
        }
    }
}

test "axpy - with alpha = -1 (subtraction) for signed types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{4};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const SubType = core.types.getType(T);
            const is_signed = switch (@typeInfo(SubType)) {
                .int => |int_info| int_info.signedness == .signed,
                .float => true,
                else => false,
            };

            // Only test signed types for subtraction
            if (is_signed) {
                const x = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer x.release(pipeline);

                const y = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer y.release(pipeline);

                const buffer_size = shape[0];

                // Initialize x with [1, 2, 3, 4]
                const x_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(x_buffer);

                for (x_buffer, 0..) |*val, i| {
                    if (comptime core.types.isComplex(T)) {
                        val.* = switch (@typeInfo(core.types.getType(T))) {
                            .float => .{ .real = @floatFromInt(i + 1), .imag = 0 },
                            .int => .{ .real = @intCast(i + 1), .imag = 0 },
                            else => unreachable,
                        };
                    } else {
                        val.* = switch (@typeInfo(T)) {
                            .float => @floatFromInt(i + 1),
                            .int => @intCast(i + 1),
                            else => unreachable,
                        };
                    }
                }

                try memory.readFromBuffer(T, pipeline, x, x_buffer);

                // Initialize y with [10, 10, 10, 10]
                const y_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(y_buffer);

                for (y_buffer) |*val| {
                    if (comptime core.types.isComplex(T)) {
                        val.* = switch (@typeInfo(core.types.getType(T))) {
                            .float => .{ .real = 10.0, .imag = 0 },
                            .int => .{ .real = 10, .imag = 0 },
                            else => unreachable,
                        };
                    } else {
                        val.* = switch (@typeInfo(T)) {
                            .float => 10.0,
                            .int => 10,
                            else => unreachable,
                        };
                    }
                }

                try memory.readFromBuffer(T, pipeline, y, y_buffer);

                // Perform y = -1*x + y (subtraction)
                const alpha: T = if (comptime core.types.isComplex(T))
                    .{ .real = -1, .imag = 0 }
                else
                    -1;

                try axpy(T, pipeline, x, alpha, y);

                // Read result
                const result_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(result_buffer);

                try memory.writeToBuffer(T, pipeline, y, result_buffer);
                pipeline.waitAndCleanup();

                // Expected: [9, 8, 7, 6] = -1*[1, 2, 3, 4] + [10, 10, 10, 10]
                for (result_buffer, 0..) |val, i| {
                    const expected_val = 10 - (@as(i64, @intCast(i)) + 1);

                    if (comptime core.types.isComplex(T)) {
                        const expected = switch (@typeInfo(core.types.getType(T))) {
                            .float => @as(core.types.getType(T), @floatFromInt(expected_val)),
                            .int => @as(core.types.getType(T), @intCast(expected_val)),
                            else => unreachable,
                        };
                        try testing.expectEqual(expected, val.real);
                        try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                    } else {
                        const expected = switch (@typeInfo(T)) {
                            .float => @as(T, @floatFromInt(expected_val)),
                            .int => @as(T, @intCast(expected_val)),
                            else => unreachable,
                        };
                        try testing.expectEqual(expected, val);
                    }
                }
            }
        }
    }
}

test "axpy - 2D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const x = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const y = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer y.release(pipeline);

            const buffer_size = shape[0] * shape[1];

            // Initialize x
            const x_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(x_buffer);

            for (x_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i), .imag = @floatFromInt(i * 2) },
                        .int => .{ .real = @intCast(i), .imag = @intCast(i * 2) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i),
                        .int => @intCast(i),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, x, x_buffer);

            // Initialize y
            const y_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(y_buffer);

            for (y_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 3), .imag = @floatFromInt(i) },
                        .int => .{ .real = @intCast(i * 3), .imag = @intCast(i) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 3),
                        .int => @intCast(i * 3),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, y, y_buffer);

            // Perform y = 3*x + y
            const alpha: T = if (comptime core.types.isComplex(T))
                .{ .real = 3, .imag = 0 }
            else
                3;

            try axpy(T, pipeline, x, alpha, y);

            // Read result
            const result_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_buffer);

            try memory.writeToBuffer(T, pipeline, y, result_buffer);
            pipeline.waitAndCleanup();

            // Verify results
            for (result_buffer, 0..) |val, i| {
                if (comptime core.types.isComplex(T)) {
                    const expected_real = i * 3 + i * 3;
                    const expected_imag = i * 2 * 3 + i;

                    const exp_real = switch (@typeInfo(core.types.getType(T))) {
                        .float => @as(core.types.getType(T), @floatFromInt(expected_real)),
                        .int => @as(core.types.getType(T), @intCast(expected_real)),
                        else => unreachable,
                    };
                    const exp_imag = switch (@typeInfo(core.types.getType(T))) {
                        .float => @as(core.types.getType(T), @floatFromInt(expected_imag)),
                        .int => @as(core.types.getType(T), @intCast(expected_imag)),
                        else => unreachable,
                    };

                    try testing.expectEqual(exp_real, val.real);
                    try testing.expectEqual(exp_imag, val.imag);
                } else {
                    const expected = i * 3 + i * 3;
                    const exp = switch (@typeInfo(T)) {
                        .float => @as(T, @floatFromInt(expected)),
                        .int => @as(T, @intCast(expected)),
                        else => unreachable,
                    };
                    try testing.expectEqual(exp, val);
                }
            }
        }
    }
}

test "axpy - 3D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 2, 2 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const x = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const y = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer y.release(pipeline);

            // Fill x with ones
            try fill.one(T, pipeline, x);

            // Fill y with value 5
            const five: T = if (comptime core.types.isComplex(T))
                .{ .real = 5, .imag = 0 }
            else
                5;

            try fill.constant(T, pipeline, y, five);

            // Perform y = 2*x + y
            const alpha: T = if (comptime core.types.isComplex(T))
                .{ .real = 2, .imag = 0 }
            else
                2;

            try axpy(T, pipeline, x, alpha, y);

            // Read result
            const buffer_size = shape[0] * shape[1] * shape[2];
            const result_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_buffer);

            try memory.writeToBuffer(T, pipeline, y, result_buffer);
            pipeline.waitAndCleanup();

            // Expected: all values should be 7 = 2*1 + 5
            for (result_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(core.types.getType(T), 7), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 7), val);
                }
            }
        }
    }
}

test "axpy - with different vector configurations" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and !(comptime core.types.isComplex(T))) {
            // Create tensors with vectors enabled
            const config_vectors = tensor_module.CreateConfig{ .vectors_enabled = true };
            const x_vec = try Tensor(T).alloc(context, pipeline, &shape, config_vectors);
            defer x_vec.release(pipeline);

            const y_vec = try Tensor(T).alloc(context, pipeline, &shape, config_vectors);
            defer y_vec.release(pipeline);

            // Create tensors with vectors disabled
            const config_no_vectors = tensor_module.CreateConfig{ .vectors_enabled = false };
            const x_no_vec = try Tensor(T).alloc(context, pipeline, &shape, config_no_vectors);
            defer x_no_vec.release(pipeline);

            const y_no_vec = try Tensor(T).alloc(context, pipeline, &shape, config_no_vectors);
            defer y_no_vec.release(pipeline);

            const buffer_size = shape[0] * shape[1];

            // Prepare input data
            const input_x = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_x);

            const input_y = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_y);

            for (input_x, input_y, 0..) |*x_val, *y_val, i| {
                x_val.* = switch (@typeInfo(T)) {
                    .float => @floatFromInt(i + 1),
                    .int => @intCast(i + 1),
                    else => unreachable,
                };
                y_val.* = switch (@typeInfo(T)) {
                    .float => @floatFromInt((i + 1) * 10),
                    .int => @intCast((i + 1) * 10),
                    else => unreachable,
                };
            }

            // Load data into all tensors
            try memory.readFromBuffer(T, pipeline, x_vec, input_x);
            try memory.readFromBuffer(T, pipeline, y_vec, input_y);
            try memory.readFromBuffer(T, pipeline, x_no_vec, input_x);
            try memory.readFromBuffer(T, pipeline, y_no_vec, input_y);

            const alpha: T = 2;

            // Perform axpy on both configurations
            try axpy(T, pipeline, x_vec, alpha, y_vec);
            try axpy(T, pipeline, x_no_vec, alpha, y_no_vec);

            // Read results
            const result_vec = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_vec);

            const result_no_vec = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_no_vec);

            try memory.writeToBuffer(T, pipeline, y_vec, result_vec);
            try memory.writeToBuffer(T, pipeline, y_no_vec, result_no_vec);
            pipeline.waitAndCleanup();

            // Both should produce the same result
            for (result_vec, result_no_vec) |v1, v2| {
                try testing.expectEqual(v1, v2);
            }
        }
    }
}

test "axpy - incompatible tensor shapes" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape_x = [_]u64{ 2, 3 };
    const shape_y = [_]u64{ 2, 4 };
    const config = tensor_module.CreateConfig{};

    const x = try Tensor(f32).alloc(context, pipeline, &shape_x, config);
    defer x.release(pipeline);

    const y = try Tensor(f32).alloc(context, pipeline, &shape_y, config);
    defer y.release(pipeline);

    const alpha: f32 = 2.0;

    // Should fail due to incompatible shapes
    const err = axpy(f32, pipeline, x, alpha, y);
    try testing.expectError(tensor_module.Errors.UnqualTensorsShape, err);
}

test "axpy - different number of dimensions" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape_2d = [_]u64{ 2, 3 };
    const shape_3d = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    const x = try Tensor(f32).alloc(context, pipeline, &shape_2d, config);
    defer x.release(pipeline);

    const y = try Tensor(f32).alloc(context, pipeline, &shape_3d, config);
    defer y.release(pipeline);

    const alpha: f32 = 1.0;

    // Should fail due to different dimensions
    const err = axpy(f32, pipeline, x, alpha, y);
    try testing.expectError(tensor_module.Errors.UnqualTensorsShape, err);
}

test "axpy - zero alpha" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const x = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const y = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer y.release(pipeline);

            // Fill x with ones
            try fill.one(T, pipeline, x);

            // Fill y with value 10
            const ten: T = if (comptime core.types.isComplex(T))
                .{ .real = 10, .imag = 0 }
            else
                10;

            try fill.constant(T, pipeline, y, ten);

            // Perform y = 0*x + y (should keep y unchanged)
            const alpha: T = if (comptime core.types.isComplex(T))
                .{ .real = 0, .imag = 0 }
            else
                0;

            try axpy(T, pipeline, x, alpha, y);

            // Read result
            const buffer_size = shape[0];
            const result_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_buffer);

            try memory.writeToBuffer(T, pipeline, y, result_buffer);
            pipeline.waitAndCleanup();

            // All values should still be 10
            for (result_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(core.types.getType(T), 10), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 10), val);
                }
            }
        }
    }
}

test "axpy - complex numbers with non-zero imaginary alpha" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{3};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and comptime core.types.isComplex(T)) {
            const x = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const y = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer y.release(pipeline);

            const buffer_size = shape[0];

            // Initialize x with complex values
            const x_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(x_buffer);

            for (x_buffer, 0..) |*val, i| {
                val.* = switch (@typeInfo(core.types.getType(T))) {
                    .float => .{ .real = @floatFromInt(i + 1), .imag = @floatFromInt(i + 1) },
                    .int => .{ .real = @intCast(i + 1), .imag = @intCast(i + 1) },
                    else => unreachable,
                };
            }

            try memory.readFromBuffer(T, pipeline, x, x_buffer);

            // Initialize y with complex values
            const y_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(y_buffer);

            for (y_buffer, 0..) |*val, i| {
                val.* = switch (@typeInfo(core.types.getType(T))) {
                    .float => .{ .real = @floatFromInt(i * 2), .imag = @floatFromInt(i * 3) },
                    .int => .{ .real = @intCast(i * 2), .imag = @intCast(i * 3) },
                    else => unreachable,
                };
            }

            try memory.readFromBuffer(T, pipeline, y, y_buffer);

            // Perform y = (2+i)*x + y
            const alpha: T = switch (@typeInfo(core.types.getType(T))) {
                .float => .{ .real = 2.0, .imag = 1.0 },
                .int => .{ .real = 2, .imag = 1 },
                else => unreachable,
            };

            try axpy(T, pipeline, x, alpha, y);

            // Read result
            const result_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_buffer);

            try memory.writeToBuffer(T, pipeline, y, result_buffer);
            pipeline.waitAndCleanup();

            // Verify: (2+i)*(a+bi) = (2a-b) + (a+2b)i
            for (result_buffer, 0..) |val, i| {
                const x_real = i + 1;
                const x_imag = i + 1;
                const y_real = i * 2;
                const y_imag = i * 3;

                // (2+i)*(x_real+x_imag*i) = (2*x_real - x_imag) + (x_real + 2*x_imag)i
                const expected_real = (2 * x_real - x_imag) + y_real;
                const expected_imag = (x_real + 2 * x_imag) + y_imag;

                const exp_real = switch (@typeInfo(core.types.getType(T))) {
                    .float => @as(core.types.getType(T), @floatFromInt(expected_real)),
                    .int => @as(core.types.getType(T), @intCast(expected_real)),
                    else => unreachable,
                };
                const exp_imag = switch (@typeInfo(core.types.getType(T))) {
                    .float => @as(core.types.getType(T), @floatFromInt(expected_imag)),
                    .int => @as(core.types.getType(T), @intCast(expected_imag)),
                    else => unreachable,
                };

                try testing.expectEqual(exp_real, val.real);
                try testing.expectEqual(exp_imag, val.imag);
            }
        }
    }
}
