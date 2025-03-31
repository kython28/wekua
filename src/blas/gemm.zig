const std = @import("std");
const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;
const Context = core.Context;

const w_tensor = @import("../tensor/main.zig");
const Tensor = w_tensor.Tensor;

const gemm_cl_kernel: []const u8 = @embedFile("kernels/gemm.cl");

pub const Operation = enum(u8) {
    no_transpose = 0,
    transpose = 1,
    // ConjugateTranspose // TODO
};

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
    is_complex: bool,
    has_alpha: bool,
    has_beta: bool,
    op_a: Operation,
    op_b: Operation,
) !cl.kernel.cl_kernel {
    const kernels_set = try KernelsSet.getKernelSet(
        command_queue,
        .GEMM,
        core.SupportedTypes.len * 2 * 2 * 2 * 2 * 2 * 2,
    );

    var kernel_index: usize = @intFromBool(vectors_enabled) * (2 * 2 * 2 * 2 * 2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(is_complex) * (2 * 2 * 2 * 2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(has_alpha) * (2 * 2 * 2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(has_beta) * (2 * 2 * core.SupportedTypes.len);
    kernel_index += @intFromEnum(op_a) * (2 * core.SupportedTypes.len);
    kernel_index += @intFromEnum(op_b) * core.SupportedTypes.len;
    kernel_index += @as(usize, core.getTypeId(T));

    if (kernels_set.kernels[kernel_index]) |v| return v;

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    const allocator = command_queue.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(
        allocator,
        "-DHAS_ALPHA={d} -DHAS_BETA={d} -DA_TRANS={d} -DB_TRANS={d}",
        .{
            @intFromBool(has_alpha),
            @intFromBool(has_beta),
            @intFromEnum(op_a),
            @intFromEnum(op_b),
        },
    );
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .is_complex = is_complex,
            .vectors_enabled = vectors_enabled,
            .kernel_name = "gemm",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        gemm_cl_kernel,
    );

    kernels_set.kernels[kernel_index] = kernel;
    kernels_set.programs[kernel_index] = program;

    return kernel;
}

inline fn validateTensors(
    comptime T: type,
    a: *Tensor(T),
    b: *Tensor(T),
    c: *Tensor(T),
    op_a: Operation,
    op_b: Operation,
) !void {
    const a_shape = a.dimensions.shape;
    const b_shape = b.dimensions.shape;
    const c_shape = c.dimensions.shape;

    if (c_shape.len != 2 or a_shape.len != 2 or b_shape.len != 2) {
        return w_tensor.Errors.InvalidValue;
    }

    const a_m = a_shape[0];
    const a_k = a_shape[1];

    const b_k = b_shape[0];
    const b_n = b_shape[1];

    const c_m = c_shape[0];
    const c_n = c_shape[1];

    const match = switch (op_a) {
        .transpose => switch (op_b) {
            .no_transpose => (a_m == b_k and b_n == c_n and a_k == c_m),
            .transpose => (a_m == b_n and b_k == c_n and a_k == c_m),
        },
        .no_transpose => switch (op_b) {
            .no_transpose => (a_k == b_k and b_n == c_n and a_m == c_m),
            .transpose => (a_k == b_n and b_k == c_n and a_m == c_m),
        },
    };

    if (!match) {
        return w_tensor.Errors.InvalidValue;
    }
}

pub fn perform(
    comptime T: type,
    command_queue: *const CommandQueue,
    alpha: ?T,
    ialpha: ?T,
    a: *Tensor(T),
    op_a: Operation,
    b: *Tensor(T),
    op_b: Operation,
    beta: ?T,
    ibeta: ?T,
    c: *Tensor(T),
) !void {
    try validateTensors(T, a, b, c, op_a, op_b);

    var real_alpha_scalar: T = undefined;
    var imag_alpha_scalar: T = undefined;

    var has_alpha = (alpha != null or ialpha != null);
    if (ialpha) |s| {
        real_alpha_scalar = alpha orelse 0;
        imag_alpha_scalar = s;
    } else {
        real_alpha_scalar = alpha orelse 1;
        imag_alpha_scalar = ialpha orelse 0;
    }

    const has_beta = (beta != null or ibeta != null);
    if (has_beta) has_alpha = true;
    const real_beta_scalar = beta orelse 0;
    const imag_beta_scalar = ibeta orelse 0;

    const vectors_enabled = (
        a.flags.vectors_enabled
        and b.flags.vectors_enabled
        and op_a == .no_transpose
        and op_b == .transpose
        and command_queue.vector_widths[Context.getTypeId(T)] > 1
    );
    const is_complex = a.flags.is_complex;

    const kernel = try getKernel(
        T,
        command_queue,
        vectors_enabled,
        is_complex,
        has_alpha,
        has_beta,
        op_a,
        op_b,
    );

    const cmd = command_queue.cmd;
    const allocator = command_queue.allocator;

    const a_prev_events = a.events_manager.getPrevEvents(.read);
    const b_prev_events = b.events_manager.getPrevEvents(.read);
    const c_prev_events = c.events_manager.getPrevEvents(.partial_write);

    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ a_prev_events, b_prev_events, c_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    var a_row_pitch: u64 = undefined;
    var b_row_pitch: u64 = undefined;
    var cols: u64 = undefined;

    if (vectors_enabled) {
        a_row_pitch = a.memory_layout.row_pitch_for_vectors;
        b_row_pitch = b.memory_layout.row_pitch_for_vectors;

        cols = a_row_pitch;
    } else {
        a_row_pitch = a.memory_layout.row_pitch;
        b_row_pitch = b.memory_layout.row_pitch;

        cols = a.dimensions.shape[1 - @intFromEnum(op_a)];
        cols += cols % 2;
        cols *= (1 + @as(u64, @intFromBool(is_complex)));
    }

    const c_row_pitch = c.memory_layout.row_pitch / (1 + @as(u64, @intFromBool(vectors_enabled)));

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    const a_buffer = &a.buffer;
    const b_buffer = &b.buffer;
    const c_buffer = &c.buffer;

    var offset: [2]u64 = .{ 0, 0 };

    const wekua_id = command_queue.wekua_id;
    const global_work_items_gemm: []const u64 = &c.work_configuration.global_work_items_gemm[wekua_id];
    const local_work_items_gemm: []const u64 = &c.work_configuration.local_work_items_gemm[wekua_id];

    const offset_y_inc = global_work_items_gemm[0];
    const offset_x_inc = global_work_items_gemm[1];

    // std.log.warn("gemm: offset_x_inc: {d}, offset_y_inc: {d}", .{ offset_x_inc, offset_y_inc });
    // std.log.warn("width: {d}, height: {d}", .{ c.work_configuration.gemm_blocks_width[wekua_id], c.work_configuration.gemm_blocks_height[wekua_id] });
    // std.log.warn("GI: {any} | LI: {any}", .{ global_work_items_gemm, local_work_items_gemm });
    // std.log.warn("Shape: {any}", .{ c.dimensions.shape });


    const blocks_height = c.work_configuration.gemm_blocks_height[wekua_id];
    const blocks_width = c.work_configuration.gemm_blocks_width[wekua_id];

    const total_kernel_events: usize = blocks_width * blocks_height;
    var counter: usize = 0;

    for (0..blocks_height) |_| {
        offset[1] = 0;
        for (0..blocks_width) |_| {
            counter += 1;

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(a_buffer));
            try set_arg(kernel, 1, cl_mem_size, @ptrCast(b_buffer));
            try set_arg(kernel, 2, cl_mem_size, @ptrCast(c_buffer));

            try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&a_row_pitch));
            try set_arg(kernel, 4, @sizeOf(u64), @ptrCast(&b_row_pitch));
            try set_arg(kernel, 5, @sizeOf(u64), @ptrCast(&c_row_pitch));

            try set_arg(kernel, 6, @sizeOf(u64), @ptrCast(&cols));

            if (has_alpha) {
                try set_arg(kernel, 7, @sizeOf(T), @ptrCast(&real_alpha_scalar));
                var arg_index: u32 = 8;
                if (has_beta) {
                    try set_arg(kernel, arg_index, @sizeOf(T), @ptrCast(&real_beta_scalar));
                    arg_index += 1;
                }

                if (is_complex) {
                    try set_arg(kernel, arg_index, @sizeOf(T), @ptrCast(&imag_alpha_scalar));
                    if (has_beta) {
                        try set_arg(kernel, arg_index + 1, @sizeOf(T), @ptrCast(&imag_beta_scalar));
                    }
                }
            }

            var new_event: cl.event.cl_event = undefined;
            try cl.kernel.enqueue_nd_range(
                cmd,
                kernel,
                &offset,
                global_work_items_gemm,
                local_work_items_gemm,
                prev_events,
                &new_event,
            );
            errdefer |err| w_tensor.helpers.releaseEvent(new_event, err);

            _ = try events_set.appendNewEvent(
                T,
                (counter == total_kernel_events),
                &.{ .read, .read, .partial_write },
                &.{ a, b, c },
                prev_events,
                new_event,
            );
            offset[1] += offset_x_inc;
        }
        offset[0] += offset_y_inc;
    }
}
