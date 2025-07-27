const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const helpers = @import("helpers.zig");

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

const fill_cl_kernel: []const u8 = @embedFile("kernels/fill.cl");

pub fn constant(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
    real_scalar: ?T,
    imag_scalar: ?T,
) !void {
    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        tensor,
        .Fill,
        "fill",
        fill_cl_kernel,
        null,
    );
    const cmd = command_queue.cmd;

    const real_scalar_value = real_scalar orelse 0;
    const imag_scalar_value = imag_scalar orelse 0;

    const prev_events = tensor.events.getPrevEvents(.write);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try set_arg(kernel, 1, @sizeOf(u64), @ptrCast(&tensor.memory_layout.row_pitch));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.memory_layout.slice_pitch));
    try set_arg(kernel, 3, @sizeOf(T), @ptrCast(&real_scalar_value));
    if (tensor.flags.is_complex) {
        try set_arg(kernel, 4, @sizeOf(T), @ptrCast(&imag_scalar_value));
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

    _ = try tensor.events.appendNewEvent(.write, prev_events, new_event, null);
}

pub inline fn one(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
) !void {
    try constant(T, command_queue, tensor, @as(T, 1), null);
}

pub inline fn zeroes(
    comptime T: type,
    tensor: *Tensor(T),
    command_queue: *const CommandQueue,
) !void {
    try constant(T, command_queue, tensor, null, null);
}
