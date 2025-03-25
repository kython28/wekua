const cl = @import("opencl");

const core = @import("../core/main.zig");
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

const helpers = @import("helpers.zig");

const transpose_cl_kernel: []const u8 = @embedFile("kernels/transpose.cl");

pub fn transpose(
    comptime T: type,
    command_queue: *const CommandQueue,
    result_tensor: *Tensor(T),
    tensor: *Tensor(T),
    dim0: u64,
    dim1: u64,
) !void {
    try helpers.eqlNumberSpace(T, tensor, result_tensor);

    const shape_a = result_tensor.shape;
    const shape_b = tensor.shape;
    if (shape_a.len != shape_b.len) {
        return w_tensor.Errors.UnqualTensorsDimension;
    } else if (dim0 >= shape_a.len or dim1 >= shape_a.len) {
        return w_tensor.Errors.InvalidValue;
    } else if (tensor.number_of_elements_without_padding != result_tensor.number_of_elements_without_padding) {
        return w_tensor.Errors.UnqualTensorsDimension;
    } else if (shape_a[dim0] != shape_b[dim1] or shape_a[dim1] != shape_b[dim0]) {
        return w_tensor.Errors.InvalidValue;
    }

    if (dim0 == dim1) {
        try w_tensor.memory.copy(T, command_queue, tensor, result_tensor);
        return;
    }

    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        tensor,
        .Transpose,
        "transpose",
        transpose_cl_kernel,
        null,
    );
    const cmd = command_queue.cmd;

    const src_prev_events = tensor.events_manager.getPrevEvents(.read);
    const dst_prev_events = result_tensor.events_manager.getPrevEvents(.write);

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

    const transpose_prev_events = try helpers.createPrevEventsResource(allocator, prev_events);
    errdefer {
        if (transpose_prev_events) |v| allocator.destroy(v);
    }

    const set_arg = cl.kernel.set_arg;
    const u64_size = @sizeOf(u64);
    const cl_mem_size = @sizeOf(cl.buffer.cl_mem);
    const shape = tensor.shape;
    const ndim: u64 = @intCast(shape.len);
    var dim0_: u64 = undefined;
    var dim1_: u64 = undefined;

    if (dim0 > dim1) {
        dim0_ = dim1;
        dim1_ = dim0;
    } else {
        dim0_ = dim0;
        dim1_ = dim1;
    }

    try set_arg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try set_arg(kernel, 1, cl_mem_size, @ptrCast(&tensor.pitchs_buffer));

    try set_arg(kernel, 2, cl_mem_size, @ptrCast(&result_tensor.buffer));
    try set_arg(kernel, 3, cl_mem_size, @ptrCast(&result_tensor.pitchs_buffer));

    try set_arg(kernel, 4, u64_size, @ptrCast(&tensor.row_pitch));
    try set_arg(kernel, 5, u64_size, @ptrCast(&tensor.shape_like_matrix_without_vectors[1]));

    try set_arg(kernel, 6, u64_size, @ptrCast(&dim0_));
    try set_arg(kernel, 7, u64_size, @ptrCast(&dim1_));
    try set_arg(kernel, 8, u64_size, @ptrCast(&ndim));

    // TODO: Adapt code to use views
    var new_event: cl.event.cl_event = undefined;
    try cl.kernel.enqueue_nd_range(
        cmd,
        kernel,
        null,
        &[1]u64{tensor.number_of_elements},
        tensor.work_item_for_all_elements[command_queue.wekua_id .. command_queue.wekua_id + 1],
        prev_events,
        &new_event,
    );
    errdefer |err| helpers.releaseEvent(new_event, err);

    try w_tensor.EventManager.appendNewEventToMultipleTensor(
        T,
        allocator,
        &.{.read, .write},
        &.{tensor, result_tensor},
        prev_events,
        new_event,
        .{ .data = transpose_prev_events, .func = &helpers.releaseEventsArray },
    );
}
