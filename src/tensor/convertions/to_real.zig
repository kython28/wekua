const std = @import("std");
const cl = @import("opencl");

const core = @import("../../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const helpers = @import("../helpers.zig");

const to_real_cl_kernel: []const u8 = @embedFile("kernels/to_real.cl");

fn get_kernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    dom: bool,
) !cl.kernel.cl_kernel {
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .ToReal, core.SupportedTypes.len * 2);
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
            .kernel_name = "to_real",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        to_real_cl_kernel,
    );

    kernels_set.kernels[index] = kernel;
    kernels_set.programs[index] = program;

    return kernel;
}

pub fn to_real(
    comptime T: type,
    command_queue: *const CommandQueue,
    src: *Tensor(T),
    dst: *Tensor(T),
    dom: bool,
) !void {
    try helpers.eqlTensorsDimensions(T, src, dst);
    if (!src.is_complex) return w_tensor.Errors.InvalidValue;
    if (dst.is_complex) return w_tensor.Errors.InvalidValue;

    const kernel = try get_kernel(T, command_queue, dom);
    const cmd = command_queue.cmd;

    const src_prev_events = src.events_manager.getPrevEvents(.read);
    const dst_prev_events = dst.events_manager.getPrevEvents(.write);

    const allocator = command_queue.allocator;
    const prev_events = try w_tensor.EventManager.concat(
        allocator,
        &.{
            src_prev_events,
            dst_prev_events,
        },
    );
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const convertion_resources = try helpers.createPrevEventsResource(allocator, prev_events);
    errdefer {
        if (convertion_resources) |v| allocator.destroy(v);
    }

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&src.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&dst.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&src.row_pitch));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&dst.row_pitch));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &src.shape_like_matrix_without_vectors,
        &src.work_items_for_matrix_shape_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| helpers.releaseEvent(new_event, err);

    try w_tensor.EventManager.appendNewEventToMultipleTensor(
        T,
        allocator,
        &.{ .read, .write },
        &.{ src, dst },
        prev_events,
        new_event,
        .{ .data = convertion_resources, .func = &helpers.releaseEventsArray },
    );
}
