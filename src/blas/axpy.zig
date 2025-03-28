const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const w_tensor = @import("../tensor/main.zig");
const Tensor = w_tensor.Tensor;

const axpy_cl_kernel: []const u8 = @embedFile("kernels/axpy.cl");

fn axpy_with_vectors(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *Tensor(T),
    alpha: ?T,
    beta: ?T,
    y: *Tensor(T),
) !void {
    const real_scalar = alpha orelse 0;
    const imag_scalar = beta orelse 0;

    const allocator = command_queue.allocator;
    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        x,
        .AXPY,
        "axpy",
        axpy_cl_kernel,
        null,
    );

    const cmd = command_queue.cmd;

    const x_prev_events = x.events_manager.getPrevEvents(.read);
    const y_prev_events = y.events_manager.getPrevEvents(.write);

    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ x_prev_events, y_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    var global: u64 = x.number_of_vectors;
    var work_items: u64 = x.local_work_items_for_vectors_1d[command_queue.wekua_id];

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, @sizeOf(T), &real_scalar);
    if (x.is_complex) {
        global /= 2;
        work_items /= 2;
        try set_arg(kernel, 3, @sizeOf(T), &imag_scalar);
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        @as([*]const u64, @ptrCast(&global))[0..1],
        @as([*]const u64, @ptrCast(&work_items))[0..1],
        prev_events,
        &new_event,
    );
    errdefer |err| w_tensor.helpers.releaseEvent(new_event, err);

    try events_set.appendNewEvent(T, &.{ .read, .write }, &.{ x, y }, prev_events, new_event);
}

fn axpy_without_vectors(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *Tensor(T),
    alpha: ?T,
    beta: ?T,
    y: *Tensor(T),
) !void {
    const real_scalar = alpha orelse 0;
    const imag_scalar = beta orelse 0;

    const allocator = command_queue.allocator;
    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        x,
        .AXPY2,
        "axpy2",
        axpy_cl_kernel,
        null,
    );

    const cmd = command_queue.cmd;

    const x_prev_events = x.events_manager.getPrevEvents(.read);
    const y_prev_events = y.events_manager.getPrevEvents(.write);

    const events_set = try w_tensor.EventManager.EventsSet.init(
        allocator,
        &.{ x_prev_events, y_prev_events },
        null,
    );
    errdefer events_set.release();

    const prev_events = events_set.getPrevEvents();

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), &x.row_pitch);
    try set_arg(kernel, 3, @sizeOf(u64), &x.slice_pitch);
    try set_arg(kernel, 4, @sizeOf(u64), &y.row_pitch);
    try set_arg(kernel, 5, @sizeOf(u64), &y.slice_pitch);
    try set_arg(kernel, 6, @sizeOf(T), &real_scalar);
    if (x.is_complex) {
        try set_arg(kernel, 7, @sizeOf(T), &imag_scalar);
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &x.global_work_items_without_vectors,
        &x.local_work_items_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| w_tensor.helpers.releaseEvent(new_event, err);

    try events_set.appendNewEvent(T, &.{ .read, .write }, &.{ x, y }, prev_events, new_event);
}

pub inline fn axpy(
    comptime T: type,
    command_queue: *const CommandQueue,
    x: *Tensor(T),
    alpha: ?T,
    beta: ?T,
    y: *Tensor(T),
) !void {
    try w_tensor.helpers.eqlTensorsDimensions(T, x, y);
    try w_tensor.helpers.eqlNumberSpace(T, x, y);
    if (alpha == null and beta == null) return w_tensor.Errors.InvalidValue;

    if (x.vectors_enabled and y.vectors_enabled) {
        try axpy_with_vectors(T, command_queue, x, alpha, beta, y);
    } else {
        try axpy_without_vectors(T, command_queue, x, alpha, beta, y);
    }
}
