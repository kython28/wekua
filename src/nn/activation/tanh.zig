const wekua = @import("../../wekua.zig");
const cl = @import("opencl");

const core = wekua.core;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const Tensor = wekua.Tensor;

const tanh_cl_kernel: []const u8 = @embedFile("kernels/tanh.cl");

pub fn Tanh(comptime T: type) type {
    const ActivationTensor = Tensor(T);

    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError("Tanh activation only supports f32 and f64 types"),
    }

    return struct {
        pub inline fn run(
            _: *const anyopaque,
            command_queue: *const CommandQueue,
            net_output: *ActivationTensor,
        ) !void {
            try wekua.math.trig.tanh(T, command_queue, net_output);
        }

        pub fn getDerivative(
            _: *const anyopaque,
            command_queue: *const CommandQueue,
            input: *ActivationTensor,
            derivative: *ActivationTensor,
        ) !void {
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                input,
                .TanhDev,
                "tanh_dev",
                tanh_cl_kernel,
                null,
            );

            const input_prev_events = input.events_manager.getPrevEvents(.read);
            const derivative_prev_events = derivative.events_manager.getPrevEvents(.write);

            const events_set = try wekua.tensor.EventManager.EventsSet.init(
                command_queue.allocator,
                &.{ input_prev_events, derivative_prev_events },
                null,
            );
            errdefer events_set.release();

            const prev_events = events_set.getPrevEvents();

            const set_arg = cl.kernel.set_arg;
            const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(&input_prev_events.buffer));
            try set_arg(kernel, 1, cl_mem_size, @ptrCast(&derivative.buffer));
            try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&input.memory_layout.row_pitch_for_vectors));
            try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&input.memory_layout.slice_pitch_for_vectors));

            var new_event: cl.event.cl_event = undefined;
            try cl.kernel.enqueue_nd_range(
                command_queue.cmd,
                kernel,
                null,
                &input.work_configuration.global_work_items,
                &input.work_configuration.local_work_items[command_queue.wekua_id],
                prev_events,
                &new_event,
            );
            errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

            try events_set.appendNewEvent(
                T,
                true,
                &.{ .read, .write },
                &.{ input, derivative },
                prev_events,
                new_event,
            );
        }
    };
}
