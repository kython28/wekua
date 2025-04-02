const std = @import("std");
const cl = @import("opencl");

const core = @import("../../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const helpers = @import("../helpers.zig");

const to_complex_cl_kernel: []const u8 = @embedFile("kernels/to_complex.cl");

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    dom: bool,
) !cl.kernel.cl_kernel {
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .ToComplex, core.SupportedTypes.len * 2);
    const index: usize = 2 * @as(usize, core.getTypeId(T)) + @intFromBool(dom);
    if (kernels_set.kernels[index]) |v| return v;

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    const allocator = command_queue.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(allocator, "-DOFFSET={d}", .{@intFromBool(dom)});
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .is_complex = false,
            .vectors_enabled = false,
            .kernel_name = "to_complex",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        to_complex_cl_kernel,
    );

    kernels_set.kernels[index] = kernel;
    kernels_set.programs[index] = program;

    return kernel;
}

pub fn toComplex(
    comptime T: type,
    command_queue: *const CommandQueue,
    src: *Tensor(T),
    dst: *Tensor(T),
    dom: bool,
) !void {
    try helpers.eqlTensorsDimensions(T, src, dst);
    if (src.flags.is_complex) return w_tensor.Errors.InvalidValue;
    if (!dst.flags.is_complex) return w_tensor.Errors.InvalidValue;

    const kernel = try getKernel(T, command_queue, dom);
    const cmd = command_queue.cmd;

    const src_prev_events = src.events_manager.getPrevEvents(.read);
    const dst_prev_events = dst.events_manager.getPrevEvents(.write);

    const allocator = command_queue.allocator;

    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ src_prev_events, dst_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&src.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&dst.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&src.memory_layout.row_pitch));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&src.memory_layout.slice_pitch));
    try set_arg(kernel, 4, @sizeOf(u64), @ptrCast(&dst.memory_layout.row_pitch));
    try set_arg(kernel, 5, @sizeOf(u64), @ptrCast(&dst.memory_layout.slice_pitch));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &src.work_configuration.global_work_items_without_vectors,
        &src.work_configuration.local_work_items_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| helpers.releaseEvent(new_event, err);

    _ = try events_set.appendNewEvent(T, true, &.{ .read, .write }, &.{ src, dst }, prev_events, new_event);
}
