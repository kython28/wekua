const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const math = @import("math");

const cache_module = @import("../layer/cache.zig");

const mse_cl_kernel: []const u8 = @embedFile("kernels/mse.cl");

fn getKernel(
    comptime T: type,
    comptime calculate_derivative: bool,
    command_queue: *const core.CommandQueue,
    vectors_enabled: bool,
) TensorErrors!cl.kernel.Kernel {
    const SUPPORTED_TYPES = core.types.SUPPORTED_TYPES;
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .MSE, SUPPORTED_TYPES.len * 2 * 2);

    var kernel_index: usize = @intFromBool(vectors_enabled) * (2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromBool(calculate_derivative) * SUPPORTED_TYPES.len;
    kernel_index += @as(usize, core.types.getTypeIndex(T));

    if (kernels_set.kernels.?[kernel_index]) |v| return v;

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    const allocator = command_queue.context.allocator;
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
    pipeline: *Pipeline,
    output: *Tensor(T),
    expected: *Tensor(T),
    cache: *const cache_module.Cache(T),
    error_result: ?*T,
) TensorErrors!void {
    const error_tensor = cache.error_tensor;

    try tensor_module.helpers.eqlTensors(T, output, expected);
    try tensor_module.helpers.eqlTensors(T, error_tensor, output);

    const command_queue = pipeline.command_queue;

    const vectors_enabled = output.flags.vectors_enabled and expected.flags.vectors_enabled and error_tensor.flags.vectors_enabled;

    const kernel = try getKernel(
        T,
        calculate_derivative,
        command_queue,
        vectors_enabled,
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&output.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&expected.buffer));
    try setArg(kernel, 2, cl_mem_size, @ptrCast(&error_tensor.buffer));

    if (calculate_derivative) {
        const last_slot = cache.slots[cache.slots.len - 1];
        const sensitivity = last_slot.layer.getSensitivity(last_slot.cache);

        try setArg(kernel, 3, cl_mem_size, @ptrCast(&sensitivity.buffer));
    }

    var num_elements: u64 = undefined;
    var work_items: u64 = undefined;
    if (vectors_enabled) {
        num_elements = output.memory_layout.number_of_vectors;
        work_items = output.work_configuration.local_work_items_for_vectors_1d[command_queue.wekua_id];
    }else{
        num_elements = output.dimensions.number_of_elements;
        work_items = output.work_configuration.local_work_items_1d[command_queue.wekua_id];
    }

    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        &.{num_elements},
        &.{work_items},
        prev_events,
        &new_event,
    );
    errdefer tensor_module.helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});

    if (error_result == null) return;

    const mean_result = try math.basic.mean(T, pipeline, error_tensor);
    error_result.?.* = mean_result;
}
