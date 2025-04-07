const wekua = @import("../../wekua.zig");
const cl = @import("opencl");

const w_cache = @import("../layer/cache.zig");

const mse_cl_kernel: []const u8 = @embedFile("kernels/mse.cl");

pub fn mse(
    comptime T: type,
    comptime calculate_gradient: bool,
    command_queue: *const wekua.core.CommandQueue,
    output: *wekua.Tensor(T),
    expected: *wekua.Tensor(T),
    cache: w_cache.Cache(T),
    error_scal: ?*T,
    errori_scal: ?*T,
) !void {
    try wekua.tensor.helpers.eqlTensors(T, output, expected);

    const kernel = try wekua.core.KernelsSet.getClKernel(
        T,
        command_queue,
        output,
        .MSE,
        "mse_kernel",
        mse_cl_kernel,
        null,
    );

    const error_tensor = cache.error_tensor;
    try wekua.tensor.helpers.eqlTensors(T, error_tensor, output);

    const output_prev_events = output.events_manager.getPrevEvents(.read);
    const expected_prev_events = expected.events_manager.getPrevEvents(.read);
    const error_tensor_prev_events = error_tensor.events_manager.getPrevEvents(.write);

    const set_args = cl.kernel.set_arg;

    try set_args(kernel, 0, @ptrCast(&output.buffer));
    try set_args(kernel, 1, @ptrCast(&expected.buffer));
    try set_args(kernel, 2, @ptrCast(&error_tensor.buffer));

    comptime var arg_index: usize = 3;
    var tensors: [4]*wekua.Tensor(T) = .{ output, expected, error_tensor, undefined };
    var operations: [4]cl.kernel.Operation = .{ .read, .read, .write, undefined };
    var prev_events_per_tensor: [4]?[]const cl.event.cl_event = .{
        output_prev_events,
        expected_prev_events,
        error_tensor_prev_events,
        null,
    };

    if (calculate_gradient) {
        const last_slot = cache.slots[cache.slots.len - 1];
        const gradient = last_slot.layer.getGradient(last_slot.cache);

        const gradient_prev_events = gradient.events_manager.getPrevEvents(.write);

        tensors[3] = gradient;
        operations[3] = .write;
        prev_events_per_tensor[3] = gradient_prev_events;

        try set_args(kernel, arg_index, @ptrCast(&gradient.buffer));
        arg_index += 1;
    }

    try set_args(kernel, arg_index, @ptrCast(&output.memory_layout.row_pitch_for_vectors));
    try set_args(kernel, arg_index + 1, @ptrCast(&output.memory_layout.slice_pitch_for_vectors));

    {
        const events_set = try wekua.tensor.EventManager.EventsSet.init(
            command_queue.allocator,
            &prev_events_per_tensor[0..arg_index],
            null,
        );
        errdefer events_set.release();

        const prev_events = events_set.getPrevEvents();

        var new_event: cl.event.cl_event = undefined;
        try cl.kernel.enqueue_nd_range(
            command_queue.cmd,
            kernel,
            null,
            &output.work_configuration.global_work_items,
            &output.work_configuration.local_work_items[command_queue.wekua_id],
            prev_events,
            &new_event,
        );
        errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

        try events_set.appendNewEvent(
            T,
            true,
            &operations[0..arg_index],
            &tensors[0..arg_index],
            prev_events,
            new_event,
        );
    }

    if (error_scal == null and errori_scal == null) return;

    try wekua.math.basic.mean(T, command_queue, error_tensor, error_scal, errori_scal);
}
