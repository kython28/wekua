const std = @import("std");

const wekua = @import("../../wekua.zig");
const cl = @import("opencl");

const core = wekua.core;
const KernelsSet = core.KernelsSet;
const CommandQueue = core.CommandQueue;

const w_cache = @import("../layer/cache.zig");

const mse_cl_kernel: []const u8 = @embedFile("kernels/mse.cl");

fn getKernel(
    comptime T: type,
    comptime calculate_derivative: bool,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
    is_complex: bool,
) !cl.kernel.cl_kernel {
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .MSE, core.SupportedTypes.len * 2 * 2 * 2);

    var kernel_index: usize = @intFromBool(vectors_enabled) * (2 * 2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(is_complex) * (2 * core.SupportedTypes.len);
    kernel_index += @intFromBool(calculate_derivative) * core.SupportedTypes.len;
    kernel_index += @as(usize, core.getTypeId(T));

    if (kernels_set.kernels.?[kernel_index]) |v| return v;

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    const allocator = command_queue.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(
        allocator,
        "-DCALC_DEV={d}",
        .{@intFromBool(calculate_derivative)},
    );
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .is_complex = is_complex,
            .vectors_enabled = vectors_enabled,
            .kernel_name = "mse_kernel",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        mse_cl_kernel,
    );

    kernels_set.kernels.?[kernel_index] = kernel;
    kernels_set.programs.?[kernel_index] = program;

    return kernel;
}

pub fn mse(
    comptime T: type,
    comptime calculate_derivative: bool,
    command_queue: *const wekua.core.CommandQueue,
    output: *wekua.Tensor(T),
    expected: *wekua.Tensor(T),
    cache: *const w_cache.Cache(T),
    error_scal: ?*T,
    errori_scal: ?*T,
) !void {
    const error_tensor = cache.error_tensor;

    try wekua.tensor.helpers.eqlTensors(T, output, expected);
    try wekua.tensor.helpers.eqlTensors(T, error_tensor, output);

    const kernel = try getKernel(
        T,
        calculate_derivative,
        command_queue,
        output.flags.vectors_enabled,
        output.flags.is_complex,
    );

    const output_prev_events = output.events_manager.getPrevEvents(.read);
    const expected_prev_events = expected.events_manager.getPrevEvents(.read);
    const error_tensor_prev_events = error_tensor.events_manager.getPrevEvents(.write);

    const set_args = cl.kernel.set_arg;

    try set_args(kernel, 0, @sizeOf(cl.buffer.cl_mem), @ptrCast(&output.buffer));
    try set_args(kernel, 1, @sizeOf(cl.buffer.cl_mem), @ptrCast(&expected.buffer));
    try set_args(kernel, 2, @sizeOf(cl.buffer.cl_mem), @ptrCast(&error_tensor.buffer));

    comptime var arg_index: usize = 3;
    var tensors: [4]*wekua.Tensor(T) = .{ output, expected, error_tensor, undefined };
    var operations: [4]wekua.tensor.EventManager.Operation = .{ .read, .read, .write, undefined };
    var prev_events_per_tensor: [4]?[]const cl.event.cl_event = .{
        output_prev_events,
        expected_prev_events,
        error_tensor_prev_events,
        null,
    };

    if (calculate_derivative) {
        const last_slot = cache.slots[cache.slots.len - 1];
        const sensitivity = last_slot.layer.getSensitivity(last_slot.cache);

        const gradient_prev_events = sensitivity.events_manager.getPrevEvents(.write);

        tensors[3] = sensitivity;
        operations[3] = .write;
        prev_events_per_tensor[3] = gradient_prev_events;

        try set_args(kernel, arg_index, @sizeOf(cl.buffer.cl_mem), @ptrCast(&sensitivity.buffer));
        arg_index += 1;
    }

    try set_args(kernel, arg_index, @sizeOf(u64), @ptrCast(&output.memory_layout.row_pitch_for_vectors));
    try set_args(kernel, arg_index + 1, @sizeOf(u64), @ptrCast(&output.memory_layout.slice_pitch_for_vectors));

    {
        const events_set = try wekua.tensor.EventManager.EventsSet.init(
            command_queue.allocator,
            prev_events_per_tensor[0..arg_index],
            null,
        );
        errdefer events_set.release();

        const prev_events = events_set.getPrevEvents();

        var new_event: cl.event.cl_event = undefined;
        try cl.kernel.enqueue_nd_range(
            command_queue.cmd,
            kernel,
            null,
            &output.work_configuration.global_work_items,
            &output.work_configuration.local_work_items[command_queue.wekua_id],
            prev_events,
            &new_event,
        );
        errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

        try events_set.appendNewEvent(
            T,
            true,
            operations[0..arg_index],
            tensors[0..arg_index],
            new_event,
        );
    }

    if (error_scal == null and errori_scal == null) return;

    try wekua.math.basic.mean(T, command_queue, error_tensor, error_scal, errori_scal);
}
