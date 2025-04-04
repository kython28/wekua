const wekua = @import("../wekua.zig");
const cl = @import("opencl");

const core = wekua.core;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const Tensor = wekua.Tensor;

const trigonometric_cl_kernel: []const u8 = @embedFile("kernels/trig.cl");

fn genericTrigFunction(
    comptime T: type,
    kernel_name: []const u8,
    kernel_id: KernelsSet.KernelsID,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        tensor,
        kernel_id,
        kernel_name,
        trigonometric_cl_kernel,
        null,
    );

    const prev_events = tensor.events_manager.getPrevEvents(.write);

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try set_arg(kernel, 1, @sizeOf(u64), @ptrCast(&tensor.memory_layout.row_pitch_for_vectors));
    try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.memory_layout.slice_pitch_for_vectors));

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        command_queue.cmd,
        kernel,
        null,
        &tensor.work_configuration.global_work_items,
        &tensor.work_configuration.local_work_items[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

    _ = try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
}

pub inline fn sin(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    try genericTrigFunction(T, "sin_kernel", .Sin, command_queue, tensor);
}

pub inline fn cos(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    try genericTrigFunction(T, "cos_kernel", .Cos, command_queue, tensor);
}

pub inline fn tan(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    try genericTrigFunction(T, "tan_kernel", .Tan, command_queue, tensor);
}

pub inline fn sinh(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    try genericTrigFunction(T, "sinh_kernel", .Sinh, command_queue, tensor);
}

pub inline fn cosh(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    try genericTrigFunction(T, "cosh_kernel", .Cosh, command_queue, tensor);
}

pub inline fn tanh(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *Tensor(T),
) !void {
    try genericTrigFunction(T, "tanh_kernel", .Tanh, command_queue, tensor);
}
