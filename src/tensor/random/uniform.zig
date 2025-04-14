const std = @import("std");
const cl = @import("opencl");

const core = @import("../../core/main.zig");
const CommandQueue = core.CommandQueue;

const helpers = @import("../helpers.zig");

const KernelsSet = core.KernelsSet;

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const uniform_random_cl_kernel: []const u8 = @embedFile("kernels/uniform.cl");

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    range_defined: bool,
    is_complex: bool,
) !cl.kernel.cl_kernel {
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .RandomUniform, core.SupportedTypes.len * 2 * 2);
    const index: usize = @as(usize, core.getTypeId(T)) * 2 * 2 + @intFromBool(is_complex) * @as(usize, 2) + @intFromBool(range_defined);
    if (kernels_set.kernels.?[index]) |v| return v;

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    const allocator = command_queue.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(allocator, "-DRANGE_DEFINED={d}", .{@intFromBool(range_defined)});
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .is_complex = is_complex,
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
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
    seed: ?u64,
    min_value: ?T,
    max_value: ?T,
) !void {
    const range_defined = min_value != null or max_value != null;
    const kernel = try getKernel(
        T,
        command_queue,
        range_defined,
        tensor.flags.is_complex,
    );
    const cmd = command_queue.cmd;

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    const global_seed = seed orelse @as(u64, @bitCast(std.time.timestamp()));

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try set_arg(kernel, 1, @sizeOf(u64), @ptrCast(&tensor.memory_layout.row_pitch));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.memory_layout.slice_pitch));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&global_seed));

    if (range_defined) {
        const min = min_value orelse switch (@typeInfo(T)) {
            .int => std.math.minInt(T),
            .float => -std.math.floatMax(T),
            else => unreachable,
        };

        const max = max_value orelse switch (@typeInfo(T)) {
            .int => std.math.maxInt(T),
            .float => std.math.floatMax(T),
            else => unreachable,
        };

        try set_arg(kernel, 4, @sizeOf(T), @ptrCast(&min));
        try set_arg(kernel, 5, @sizeOf(T), @ptrCast(&max));
    }

    // TODO: Adapt code to use views
    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &tensor.work_configuration.global_work_items_without_vectors,
        &tensor.work_configuration.local_work_items_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| helpers.releaseEvent(new_event, err);

    _ = try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
}
