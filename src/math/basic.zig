const wekua = @import("../wekua.zig");
const cl = @import("opencl");

const core = wekua.core;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const dot_cl_kernel: []const u8 = @embedFile("kernels/dot.cl");

fn genericBasicMathFunction(
    comptime T: type,
    kernel_name: []const u8,
    kernel_id: KernelsSet.KernelsID,
    kernel_source: []const u8,
    command_queue: *const CommandQueue,
    x: *const wekua.Tensor(T),
    y: *const wekua.Tensor(T),
) !void {
    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        x,
        kernel_id,
        kernel_name,
        kernel_source,
        null,
    );

    const x_prev_events = x.events_manager.getPrevEvents(.write);
    const y_prev_events = y.events_manager.getPrevEvents(.read);

    const events_set = try wekua.tensor.EventManager.EventsSet.init(
        command_queue.allocator,
        &.{ x_prev_events, y_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&x.memory_layout.row_pitch));
    try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&x.memory_layout.slice_pitch));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        command_queue.cmd,
        kernel,
        null,
        &x.work_configuration.global_work_items,
        &x.work_configuration.local_work_items[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

    _ = try events_set.appendNewEvent(T, true, &.{ .write, .read }, &.{ x, y }, prev_events, new_event);
}

pub inline fn dot(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *const wekua.Tensor(T),
    y: *const wekua.Tensor(T),
) !void {
    try genericBasicMathFunction(
        T,
        "dot_kernel",
        .Dot,
        dot_cl_kernel,
        command_queue,
        x,
        y,
    );
}
