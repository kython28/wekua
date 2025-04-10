const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

const utils = @import("../utils/utils.zig");
const helpers = @import("helpers.zig");

const identity_cl_kernel: []const u8 = @embedFile("kernels/identity.cl");

pub fn identity(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    const size = tensor.dimensions.shape[0];
    for (tensor.dimensions.shape[1..]) |s| {
        if (s != size) {
            return w_tensor.Errors.InvalidValue;
        }
    }

    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        tensor,
        .Identity,
        "identity",
        identity_cl_kernel,
        null,
    );
    const cmd = command_queue.cmd;

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    var work_items: u64 = undefined;
    utils.calculateWorkItems(
        &.{ size },
        @as([*]u64, @ptrCast(&work_items))[0..1],
        command_queue.max_work_group_size,
    );

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&tensor.pitches_buffer));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.dimensions.shape.len));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &.{ size },
        &.{ work_items },
        prev_events,
        &new_event,
    );
    errdefer |err| helpers.releaseEvent(new_event, err);

    _ = try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
}
