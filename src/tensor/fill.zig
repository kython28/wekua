const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const helpers = @import("helpers.zig");

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const fill_cl_kernel: []const u8 = @embedFile("kernels/fill.cl");

pub fn constant(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    scalar: T,
) TensorErrors!void {
    const command_queue = pipeline.command_queue;
    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        .Fill,
        "fill",
        fill_cl_kernel,
        null,
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try setArg(kernel, 1, @sizeOf(u64), @ptrCast(&tensor.memory_layout.row_pitch));
    try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.memory_layout.slice_pitch));
    try setArg(kernel, 3, @sizeOf(T), @ptrCast(&scalar));


    // TODO: Adapt code to use views
    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        pipeline.command_queue.cl_command_queue,
        kernel,
        null,
        &tensor.work_configuration.global_work_items_without_vectors,
        &tensor.work_configuration.local_work_items_without_vectors[pipeline.command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    _ = try tensor.events.appendNewEvent(.write, prev_events, new_event, null);
}

pub inline fn one(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) !void {
    try constant(T, pipeline, tensor, @as(T, 1), null);
}

pub fn zeroes(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    const prev_events = pipeline.prevEvents();

    const zero: T = std.mem.zeroes(T);

    var new_event: cl.event.Event = undefined;
    try cl.buffer.fill(
        pipeline.command_queue.cl_command_queue,
        tensor.buffer,
        &zero,
        @sizeOf(T),
        0,
        tensor.memory_layout.size,
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}
