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
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .RandomUniform, core.SupportedTypes.len * 2);
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
    min_value: ?T,
    max_value: ?T,
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
