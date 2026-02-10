const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const utils = @import("utils");

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const dot_cl_kernel: []const u8 = @embedFile("kernels/dot.cl");
const sum_cl_kernel: []const u8 = @embedFile("kernels/sum.cl");

pub fn dot(
    comptime T: type,
    pipeline: *Pipeline,
    x: *Tensor(T),
    y: *Tensor(T),
) TensorErrors!void {
    try tensor_module.helpers.eqlTensorsShape(T, x, y);

    const command_queue = pipeline.command_queue;

    const vectors_enabled = x.flags.vectors_enabled and y.flags.vectors_enabled;
    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        vectors_enabled,
        .Dot,
        "dot_kernel",
        dot_cl_kernel,
        null,
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));

    const wekua_id = command_queue.wekua_id;
    var global_work_items: []const u64 = undefined;
    var local_work_items: []const u64 = undefined;

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

fn executeSum(
    comptime T: type,
    pipeline: *Pipeline,
    x: *Tensor(T),
    result: *Tensor(T),
) TensorErrors!void {
    const command_queue = pipeline.command_queue;

    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        x.flags.vectors_enabled,
        .Sum,
        "sum_kernel",
        sum_cl_kernel,
        null,
    );

    const prev_events = pipeline.prevEvents();

    const global_work_items: []const u64 = x.work_configuration.global_work_items[0..2];
    var local_work_items: [2]u64 = undefined;

    utils.calculateWorkItems(global_work_items, &local_work_items, command_queue.max_work_group_size);

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&result.buffer));
    try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&x.memory_layout.row_pitch_for_vectors));
    try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&x.memory_layout.slice_pitch_for_vectors));
    try setArg(kernel, 4, @sizeOf(u64), @ptrCast(&global_work_items[1]));

    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        global_work_items,
        &local_work_items,
        prev_events,
        &new_event,
    );
    errdefer tensor_module.helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

pub fn sum(
    comptime T: type,
    pipeline: *Pipeline,
    x: *Tensor(T),
) TensorErrors!T {
    const command_queue = pipeline.command_queue;
    const context = command_queue.context;

    var row_length: u64 = 1;
    for (x.dimensions.shape[0..(x.dimensions.shape.len - 1)]) |s| {
        row_length *= s;
    }

    const last_dim = x.dimensions.shape[x.dimensions.shape.len - 1];

    var temporal_tensor: *Tensor(T) = undefined;
    var temporal_tensor_allocated = false;

    if (last_dim > 1) {
        temporal_tensor = try Tensor(T).alloc(context, pipeline, &.{ 1, row_length }, .{});
        temporal_tensor_allocated = true;
        errdefer temporal_tensor.release(pipeline);

        try executeSum(T, pipeline, x, temporal_tensor);
    } else {
        temporal_tensor = x;
    }

    const prev_events = pipeline.prevEvents();

    var mapping_event: cl.event.Event = undefined;
    const buf_map = try cl.buffer.map(
        []T,
        command_queue.cl_command_queue,
        temporal_tensor.buffer,
        false,
        cl.buffer.MapFlag.read,
        0,
        @sizeOf(T) * row_length,
        prev_events,
        &mapping_event,
    );

    cl.event.wait(mapping_event) catch |err| {
        std.debug.panic("An error ocurred while waiting for mapping event ({s})", .{@errorName(err)});
    };

    defer {
        cl.buffer.unmap(
            []T,
            command_queue.cl_command_queue,
            temporal_tensor.buffer,
            buf_map,
            null,
            null,
        ) catch |err| {
            std.debug.panic("An error ocurred while unmapping buffer ({s})", .{@errorName(err)});
        };

        if (temporal_tensor_allocated) temporal_tensor.release(pipeline);
    }

    var result: T = std.mem.zeroes(T);
    if (comptime core.types.isComplex(T)) {
        for (buf_map) |v| {
            result.real += v.real;
            result.imag += v.imag;
        }
    } else {
        for (buf_map) |v| {
            result += v;
        }
    }

    return result;
}

// TODO: Add support for selecting the axis
pub fn mean(
    comptime T: type,
    pipeline: *Pipeline,
    x: *Tensor(T),
) TensorErrors!T {
    var result = try sum(T, pipeline, x);

    const number_of_elements = x.dimensions.number_of_elements_without_padding;
    const SubType = core.types.getType(T);

    if (comptime core.types.isComplex(T)) {
        switch (@typeInfo(SubType)) {
            .int => {
                result.real = @divTrunc(result.real, @as(SubType, @intCast(number_of_elements)));
                result.imag = @divTrunc(result.imag, @as(SubType, @intCast(number_of_elements)));
            },
            .float => {
                result.real /= @floatFromInt(number_of_elements);
                result.imag /= @floatFromInt(number_of_elements);
            },
            else => unreachable,
        }
    } else {
        switch (@typeInfo(T)) {
            .int => {
                result = @divTrunc(result, @as(T, @intCast(number_of_elements)));
            },
            .float => {
                result /= @floatFromInt(number_of_elements);
            },
            else => unreachable,
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests
const testing = std.testing;

const memory = tensor_module.memory;
const fill = tensor_module.fill;

fn castInt(comptime T: type, val: anytype) T {
    return switch (@typeInfo(T)) {
        .float => @floatFromInt(val),
        .int => @intCast(val),
        else => unreachable,
    };
}

test "dot - element-wise multiplication" {
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

            const x_buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(x_buf);

            const y_buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(y_buf);

            for (x_buf, y_buf, 0..) |*xv, *yv, i| {
                if (comptime core.types.isComplex(T)) {
                    const SubType = core.types.getType(T);
                    xv.* = .{ .real = castInt(SubType, i + 1), .imag = 0 };
                    yv.* = .{ .real = castInt(SubType, i + 2), .imag = 0 };
                } else {
                    xv.* = castInt(T, i + 1);
                    yv.* = castInt(T, i + 2);
                }
            }

            try memory.readFromBuffer(T, pipeline, x, x_buf);
            try memory.readFromBuffer(T, pipeline, y, y_buf);

            try dot(T, pipeline, x, y);

            const result = try allocator.alloc(T, shape[0]);
            defer allocator.free(result);

            try memory.writeToBuffer(T, pipeline, x, result);
            pipeline.waitAndCleanup();

            // Expected: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
            for (result, 0..) |val, i| {
                const expected = (i + 1) * (i + 2);
                if (comptime core.types.isComplex(T)) {
                    const SubType = core.types.getType(T);
                    try testing.expectEqual(castInt(SubType, expected), val.real);
                    try testing.expectEqual(@as(SubType, 0), val.imag);
                } else {
                    try testing.expectEqual(castInt(T, expected), val);
                }
            }
        }
    }
}

test "sum - basic sum operation" {
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

            const buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(buf);

            // [1, 2, 3, 4, 5] → sum = 15
            for (buf, 0..) |*v, i| {
                if (comptime core.types.isComplex(T)) {
                    const SubType = core.types.getType(T);
                    v.* = .{ .real = castInt(SubType, i + 1), .imag = 0 };
                } else {
                    v.* = castInt(T, i + 1);
                }
            }

            try memory.readFromBuffer(T, pipeline, x, buf);

            const result = try sum(T, pipeline, x);

            if (comptime core.types.isComplex(T)) {
                const SubType = core.types.getType(T);
                try testing.expectEqual(castInt(SubType, 15), result.real);
                try testing.expectEqual(@as(SubType, 0), result.imag);
            } else {
                try testing.expectEqual(castInt(T, 15), result);
            }
        }
    }
}

test "mean - basic mean operation for float types" {
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
            const is_float = comptime blk: {
                if (core.types.isComplex(T)) {
                    break :blk @typeInfo(core.types.getType(T)) == .float;
                }
                break :blk @typeInfo(T) == .float;
            };

            if (is_float) {
                const x = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer x.release(pipeline);

                const buf = try allocator.alloc(T, shape[0]);
                defer allocator.free(buf);

                // [2, 4, 6, 8] → mean = 5.0
                for (buf, 0..) |*v, i| {
                    if (comptime core.types.isComplex(T)) {
                        const SubType = core.types.getType(T);
                        v.* = .{ .real = castInt(SubType, (i + 1) * 2), .imag = 0 };
                    } else {
                        v.* = castInt(T, (i + 1) * 2);
                    }
                }

                try memory.readFromBuffer(T, pipeline, x, buf);

                const result = try mean(T, pipeline, x);

                const eps: f64 = 1e-5;
                if (comptime core.types.isComplex(T)) {
                    try testing.expectApproxEqAbs(@as(f64, 5.0), @as(f64, @floatCast(result.real)), eps);
                    try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result.imag)), eps);
                } else {
                    try testing.expectApproxEqAbs(@as(f64, 5.0), @as(f64, @floatCast(result)), eps);
                }
            }
        }
    }
}
