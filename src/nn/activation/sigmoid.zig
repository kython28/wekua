const wekua = @import("../../wekua.zig");
const cl = @import("opencl");

const activation = @import("main.zig");

const core = wekua.core;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const Tensor = wekua.Tensor;

const sigmoid_cl_kernel: []const u8 = @embedFile("kernels/sigmoid.cl");

pub fn Sigmoid(comptime T: type) type {
    const ActivationTensor = Tensor(T);

    const Activation = activation.Activation(T);

    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError("Sigmoid activation only supports f32 and f64 types"),
    }

    return struct {
        pub fn init() Activation {
            return Activation{
                .vtable = .{
                    .run = &run,
                    .getDerivative = &getDerivative,
                },
                .ptr = undefined,
            };
        }

        pub fn deinit(_: *const anyopaque) void {}

        pub fn run(_: *const anyopaque, command_queue: *const CommandQueue, net_output: *ActivationTensor) !void {
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                net_output,
                .Sigmoid,
                "sigmoid",
                sigmoid_cl_kernel,
                null,
            );

            const prev_events = net_output.events_manager.getPrevEvents(.write);

            const set_arg = cl.kernel.set_arg;
            const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(&net_output.buffer));
            try set_arg(kernel, 1, @sizeOf(u64), @ptrCast(&net_output.memory_layout.row_pitch_for_vectors));
            try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&net_output.memory_layout.slice_pitch_for_vectors));

            var new_event: cl.event.cl_event = undefined;
            try cl.kernel.enqueue_nd_range(
                command_queue.cmd,
                kernel,
                null,
                &net_output.work_configuration.global_work_items,
                &net_output.work_configuration.local_work_items[command_queue.wekua_id],
                prev_events,
                &new_event,
            );
            errdefer |err| wekua.tensor.helpers.releaseEvent(new_event, err);

            _ = try net_output.events_manager.appendNewEvent(.write, prev_events, new_event, null);
        }

        pub fn getDerivative(
            _: *const anyopaque,
            command_queue: *const CommandQueue,
            output: *ActivationTensor,
            derivative: *ActivationTensor,
        ) !void {
            try wekua.tensor.helpers.eqlTensors(T, output, derivative);

            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                output,
                .SigmoidDev,
                "sigmoid_dev",
                sigmoid_cl_kernel,
                null,
            );

            const output_prev_events = output.events_manager.getPrevEvents(.read);
            const derivative_prev_events = derivative.events_manager.getPrevEvents(.write);

            const events_set = try wekua.tensor.EventManager.EventsSet.init(
                command_queue.allocator,
                &.{ output_prev_events, derivative_prev_events },
                null,
            );
            errdefer events_set.release();

            const prev_events = events_set.getPrevEvents();

            const set_arg = cl.kernel.set_arg;
            const cl_mem_size = @sizeOf(cl.buffer.cl_mem);

            try set_arg(kernel, 0, cl_mem_size, @ptrCast(&output.buffer));
            try set_arg(kernel, 1, cl_mem_size, @ptrCast(&derivative.buffer));
            try set_arg(kernel, 2, @sizeOf(u64), @ptrCast(&output.memory_layout.row_pitch_for_vectors));
            try set_arg(kernel, 3, @sizeOf(u64), @ptrCast(&output.memory_layout.slice_pitch_for_vectors));

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
                &.{ .read, .write },
                &.{ output, derivative },
                new_event,
            );
        }
    };
}
