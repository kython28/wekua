const std = @import("std");
const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const w_tensor = @import("../tensor/main.zig");
const Tensor = w_tensor.Tensor;

const axpy_cl_kernel: []const u8 = @embedFile("kernels/axpy.cl");

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    comptime kernel_name: []const u8,
    comptime vectors_enabled: bool,
    is_complex: bool,
    has_alpha: bool,
    substract: bool,
) !cl.kernel.cl_kernel {
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .AXPY, core.SupportedTypes.len * 2 * 2 * 2 * 2);

    var kernel_index: usize = @intFromBool(vectors_enabled) * (2 * 2 * 2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(is_complex) * (2 * 2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(has_alpha) * (2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(substract) * core.SupportedTypes.len;
    kernel_index += @as(usize, core.getTypeId(T));

    if (kernels_set.kernels.?[kernel_index]) |v| return v;

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    const allocator = command_queue.allocator;
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
            .is_complex = is_complex,
            .vectors_enabled = vectors_enabled,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        axpy_cl_kernel,
    );

    kernels_set.kernels.?[kernel_index] = kernel;
    kernels_set.programs.?[kernel_index] = program;

    return kernel;
}

inline fn checkAlpha(comptime T: type, alpha: ?T, ialpha: ?T) bool {
    return (alpha != null or ialpha != null);
}

inline fn checkIfSubstract(comptime T: type, alpha: T, ialpha: T) bool {
    return switch (@typeInfo(T)) {
        .float => blk: {
            const eps = comptime std.math.floatEps(T);
            break :blk (@abs(alpha + @as(T, 1)) < eps and @abs(ialpha) < eps);
        },
        .int => |int_info| blk: {
            if (int_info.signedness == .unsigned) {
                break :blk false;
            }
            break :blk (alpha == @as(T, -1) and ialpha == 0);
        },
        else => @compileError("Type not supported"),
    };
}

fn axpyWithVectors(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *Tensor(T),
    alpha: ?T,
    ialpha: ?T,
    y: *Tensor(T),
) !void {
    var real_scalar: T = undefined;
    var imag_scalar: T = undefined;

    var has_alpha = checkAlpha(T, alpha, ialpha);
    var substract = false;
    if (has_alpha) {
        if (ialpha) |s| {
            real_scalar = alpha orelse 0;
            imag_scalar = s;
        }else{
            real_scalar = alpha orelse 1;
            imag_scalar = ialpha orelse 0;
        }

        substract = checkIfSubstract(T, real_scalar, imag_scalar);
        if (substract) {
            has_alpha = false;
        }
    }

    const allocator = command_queue.allocator;
    const kernel = try getKernel(
        T,
        command_queue,
        "axpy",
        true,
        x.flags.is_complex,
        has_alpha,
        substract
    );

    const cmd = command_queue.cmd;

    const x_prev_events = x.events_manager.getPrevEvents(.read);
    const y_prev_events = y.events_manager.getPrevEvents(.write);

    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ x_prev_events, y_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), &x.memory_layout.row_pitch_for_vectors);
    try set_arg(kernel, 3, @sizeOf(u64), &x.memory_layout.slice_pitch_for_vectors);
    if (has_alpha) {
        try set_arg(kernel, 4, @sizeOf(T), &real_scalar);
        if (x.flags.is_complex) {
            try set_arg(kernel, 5, @sizeOf(T), &imag_scalar);
        }
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &x.work_configuration.global_work_items,
        &x.work_configuration.local_work_items[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| w_tensor.helpers.releaseEvent(new_event, err);

    _ = try events_set.appendNewEvent(T, true, &.{ .read, .write }, &.{ x, y }, new_event);
}

fn axpyWithoutVectors(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *Tensor(T),
    alpha: ?T,
    ialpha: ?T,
    y: *Tensor(T),
) !void {
    var real_scalar: T = undefined;
    var imag_scalar: T = undefined;

    var has_alpha = checkAlpha(T, alpha, ialpha);
    var substract = false;
    if (has_alpha) {
        if (ialpha) |s| {
            real_scalar = alpha orelse 0;
            imag_scalar = s;
        }else{
            real_scalar = alpha orelse 1;
            imag_scalar = ialpha orelse 0;
        }

        substract = checkIfSubstract(T, real_scalar, imag_scalar);
        if (substract) {
            has_alpha = false;
        }
    }

    const allocator = command_queue.allocator;
    const kernel = try getKernel(
        T,
        command_queue,
        "axpy2",
        false,
        x.flags.is_complex,
        has_alpha,
        substract
    );

    const cmd = command_queue.cmd;

    const x_prev_events = x.events_manager.getPrevEvents(.read);
    const y_prev_events = y.events_manager.getPrevEvents(.write);

    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ x_prev_events, y_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), &x.memory_layout.row_pitch);
    try set_arg(kernel, 3, @sizeOf(u64), &x.memory_layout.slice_pitch);
    try set_arg(kernel, 4, @sizeOf(u64), &y.memory_layout.row_pitch);
    try set_arg(kernel, 5, @sizeOf(u64), &y.memory_layout.slice_pitch);

    if (has_alpha) {
        try set_arg(kernel, 6, @sizeOf(T), &real_scalar);
        if (x.flags.is_complex) {
            try set_arg(kernel, 7, @sizeOf(T), &imag_scalar);
        }
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &x.work_configuration.global_work_items_without_vectors,
        &x.work_configuration.local_work_items_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| w_tensor.helpers.releaseEvent(new_event, err);

    _ = try events_set.appendNewEvent(T, true, &.{ .read, .write }, &.{ x, y }, new_event);
}

pub inline fn axpy(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *Tensor(T),
    alpha: ?T,
    ialpha: ?T,
    y: *Tensor(T),
) !void {
    try w_tensor.helpers.eqlTensorsDimensions(T, x, y);
    try w_tensor.helpers.eqlNumberSpace(T, x, y);

    if (x.flags.vectors_enabled and y.flags.vectors_enabled) {
        try axpyWithVectors(T, command_queue, x, alpha, ialpha, y);
    } else {
        try axpyWithoutVectors(T, command_queue, x, alpha, ialpha, y);
    }
}
