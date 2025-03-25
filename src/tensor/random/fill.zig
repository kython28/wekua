const std = @import("std");
const cl = @import("opencl");

const core = @import("../../core/main.zig");
const CommandQueue = core.CommandQueue;

const helpers = @import("../helpers.zig");

const KernelsSet = core.KernelsSet;

const w_tensor = @import("../main.zig");
const Tensor = w_tensor.Tensor;

const random_cl_kernel: []const u8 = @embedFile("kernels/fill.cl");

pub fn fill(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *CommandQueue,
    seed: ?u64,
) !void {
    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        tensor,
        .FillRandom,
        "random",
        random_cl_kernel,
        null,
    );
    const cmd = command_queue.cmd;

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);
    const shape = tensor.shape;

    const global_seed = seed orelse @as(u64, @bitCast(std.time.timestamp()));

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try set_arg(kernel, 1, @sizeOf(u64), @ptrCast(&tensor.row_pitch));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&shape[shape.len - 1]));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&global_seed));

    // TODO: Adapt code to use views
    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &tensor.shape_like_matrix_without_vectors,
        &tensor.work_items_like_matrix_without_vectors[command_queue.wekua_id],
        null,
        &new_event,
    );
    errdefer |err| helpers.release_event(new_event, err);

    try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
}
