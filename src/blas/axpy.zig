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

    const prev_events = try w_tensor.EventManager.concat(allocator, &.{ x_prev_events, y_prev_events });
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const axpy_resources = try w_tensor.helpers.createPrevEventsResource(allocator, prev_events);
    errdefer {
        if (axpy_resources) |v| allocator.destroy(v);
    }

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    var global: u64 = x.number_of_vectors;
    var work_items: u64 = x.work_item_for_all_vectors[command_queue.wekua_id];

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

    try w_tensor.EventManager.appendNewEventToMultipleTensor(
        T,
        allocator,
        &.{ .read, .write },
        &.{ x, y },
        prev_events,
        new_event,
        .{ .data = axpy_resources, .func = &w_tensor.helpers.releaseEventsArray },
    );
}

fn axpy_without_vectors( comptime T: type,
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

    const prev_events = try w_tensor.EventManager.concat(allocator, &.{ x_prev_events, y_prev_events });
    errdefer {
        if (prev_events) |v| allocator.free(v);
    }

    const axpy_resources = try w_tensor.helpers.createPrevEventsResource(allocator, prev_events);
    errdefer {
        if (axpy_resources) |v| allocator.destroy(v);
    }

    const set_arg = cl.kernel.set_arg;
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&x.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&y.buffer));
    try set_arg(kernel, 2, @sizeOf(u64), &x.row_pitch);
    try set_arg(kernel, 3, @sizeOf(u64), &y.row_pitch);
    try set_arg(kernel, 4, @sizeOf(T), &real_scalar);
    if (x.is_complex) {
        try set_arg(kernel, 5, @sizeOf(T), &imag_scalar);
    }

    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &x.shape_like_matrix_without_vectors,
        &x.work_items_for_matrix_shape_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer |err| w_tensor.helpers.releaseEvent(new_event, err);

    try w_tensor.EventManager.appendNewEventToMultipleTensor(
        T,
        allocator,
        &.{ .read, .write },
        &.{ x, y },
        prev_events,
        new_event,
        .{ .data = axpy_resources, .func = &w_tensor.helpers.releaseEventsArray },
    );
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
    }else{
        try axpy_without_vectors(T, command_queue, x, alpha, beta, y);
    }
}
