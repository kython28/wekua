const wekua = @import("../../wekua.zig");
const cl = @import("opencl");

const w_cache = @import("../layer/cache.zig");

const mse_cl_kernel: []const u8 = @embedFile("kernels/mse.cl");

pub fn perform(
    comptime T: type,
    comptime calculate_gradient: bool,
    command_queue: *const wekua.core.CommandQueue,
    cache: w_cache.Cache(T),
    output: *wekua.Tensor(T),
    expected: *wekua.Tensor(T),
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
    try set_args(kernel, 3, @ptrCast(&output.memory_layout.row_pitch_for_vectors));
    try set_args(kernel, 4, @ptrCast(&output.memory_layout.slice_pitch_for_vectors));

    {
        const events_set = try wekua.tensor.EventManager.EventsSet.init(
            command_queue.allocator,
            &.{ output_prev_events, expected_prev_events, error_tensor_prev_events },
            null,
        );
        errdefer events_set.release();

        const prev_events = events_set.getPrevEvents(.write);
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
            &.{ .read, .read, .write },
            &.{ output, expected, error_tensor },
            prev_events,
            new_event,
        );
    }

    try error_tensor.wait();


}
