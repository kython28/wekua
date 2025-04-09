const std = @import("std");

const wekua = @import("../wekua.zig");
const cl = @import("opencl");

const core = wekua.core;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const dot_cl_kernel: []const u8 = @embedFile("kernels/dot.cl");
const sum_cl_kernel: []const u8 = @embedFile("kernels/sum.cl");

const helpers = wekua.tensor.helpers;

fn genericBasicMathFunction(
    comptime T: type,
    kernel_name: []const u8,
    kernel_id: KernelsSet.KernelsID,
    kernel_source: []const u8,
    command_queue: *const CommandQueue,
    x: *wekua.Tensor(T),
    y: *wekua.Tensor(T),
) !void {
    try helpers.eqlTensors(T, x, y);
    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        x,
        kernel_id,
        kernel_name,
        kernel_source,
        null,
    );

    const x_prev_events = x.events_manager.getPrevEvents(.write);
    const y_prev_events = y.events_manager.getPrevEvents(.read);

    const events_set = try wekua.tensor.EventManager.EventsSet.init(
        command_queue.allocator,
        &.{ x_prev_events, y_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&x.memory_layout.row_pitch_for_vectors));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&x.memory_layout.slice_pitch_for_vectors));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        command_queue.cmd,
        kernel,
        null,
        &x.work_configuration.global_work_items,
        &x.work_configuration.local_work_items[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

    try events_set.appendNewEvent(T, true, &.{ .write, .read }, &.{ x, y }, prev_events, new_event);
}

pub inline fn dot(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *wekua.Tensor(T),
    y: *wekua.Tensor(T),
) !void {
    try genericBasicMathFunction(
        T,
        "dot_kernel",
        .Dot,
        dot_cl_kernel,
        command_queue,
        x,
        y,
    );
}

fn executeSum(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *wekua.Tensor(T),
    result: *wekua.Tensor(T),
) !void {
    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        x,
        .Sum,
        "sum_kernel",
        sum_cl_kernel,
        null,
    );

    const x_prev_events = x.events_manager.getPrevEvents(.read);
    const result_prev_events = result.events_manager.getPrevEvents(.write);

    const events_set = try wekua.tensor.EventManager.EventsSet.init(
        command_queue.allocator,
        &.{ x_prev_events, result_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const global_work_items: []const u64 = x.work_configuration.global_work_items[0..2];
    var local_work_items: [2]u64 = undefined;

    wekua.utils.calculateWorkItems(global_work_items, &local_work_items, command_queue.max_work_group_size);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&result.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&x.memory_layout.row_pitch_for_vectors));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&x.memory_layout.slice_pitch_for_vectors));
    try set_arg(kernel, 4, @sizeOf(u64), @ptrCast(&global_work_items[1]));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        command_queue.cmd,
        kernel,
        null,
        global_work_items,
        &local_work_items,
        x_prev_events,
        &new_event,
    );
    errdefer |err| helpers.releaseEvent(new_event, err);

    try events_set.appendNewEvent(T, true, &.{ .read, .write }, &.{ x, result }, prev_events, new_event);
}

pub fn sum(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *wekua.Tensor(T),
    result: ?*T,
    imag_result: ?*T,
) !void {
    if (result == null and imag_result == null) {
        return wekua.tensor.Errors.InvalidValue;
    }

    if (x.flags.is_complex and imag_result == null) {
        return wekua.tensor.Errors.InvalidValue;
    }

    var row_length: u64 = 1;
    for (x.dimensions.shape[0..(x.dimensions.shape.len - 1)]) |s| {
        row_length *= s;
    }

    const is_complex = x.flags.is_complex;

    var temporal_tensor_allocated: bool = false;
    var temporal_tensor: *wekua.Tensor(T) = undefined;
    var prev_events: ?[]const cl.event.cl_event = null;

    if (row_length > 1) {
        temporal_tensor = try wekua.Tensor(T).alloc(x.context, &.{ 1, row_length }, .{
            .is_complex = is_complex,
        });
        temporal_tensor_allocated = true;
        errdefer temporal_tensor.release();

        try executeSum(
            T,
            command_queue,
            x,
            temporal_tensor,
        );

        prev_events = temporal_tensor.events_manager.getPrevEvents(.read);
    } else {
        temporal_tensor = x;
    }

    var new_event: cl.event.cl_event = undefined;
    const buf_map = try cl.buffer.map(
        []T,
        command_queue.cmd,
        temporal_tensor.buffer,
        false,
        @intFromEnum(cl.buffer.enums.map_flags.read),
        0,
        @sizeOf(T) * row_length * (1 + @as(u64, @intFromBool(is_complex))),
        prev_events,
        &new_event
    );
    cl.event.wait(new_event) catch |err| {
        std.debug.panic("An error ocurred while waiting for mapping event ({s})", .{@errorName(err)});
    };

    defer {
        cl.buffer.unmap(
            []T,
            command_queue.cmd,
            temporal_tensor.buffer,
            buf_map,
            null,
            null,
        ) catch |err| {
            std.debug.panic("An error ocurred while unmapping buffer ({s})", .{@errorName(err)});
        };

        if (temporal_tensor_allocated) temporal_tensor.release();
    }

    var _result: T = undefined;
    var _imag_result: T = undefined;

    if (is_complex) {
        const elements_count = row_length * 2;
        var index: usize = 0;
        while (index < elements_count) : (index += 2) {
            _result += buf_map[index];
            _imag_result += buf_map[index + 1];
        }
    }else{
        for (buf_map) |v| {
            _result += v;
        }
    }

    if (result) |res| {
        res.* = _result;
    }

    if (imag_result) |imag_res| {
        imag_res.* = _imag_result;
    }
}

// TODO: Add support for selecting the axis
pub fn mean(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *wekua.Tensor(T),
    result: ?*T,
    imag_result: ?*T,
) !void {
    try sum(T, command_queue, x, result, imag_result);

    const number_of_elements = x.dimensions.number_of_elements_without_padding / (1 + @as(u64, @intFromBool(x.flags.is_complex)));
    switch (@typeInfo(T)) {
        .int => {
            if (result) |res| {
                res.* /= @intCast(number_of_elements);
            }
            if (imag_result) |imag_res| {
                imag_res.* /= @intCast(number_of_elements);
            }
        },
        .float => {
            if (result) |res| {
                res.* /= @floatFromInt(number_of_elements);
            }
            if (imag_result) |imag_res| {
                imag_res.* /= @floatFromInt(number_of_elements);
            }
        },
        else => unreachable,
    }
}
