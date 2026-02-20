const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const helpers = tensor_module.helpers;

const math = @import("math");

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
            pipeline: *Pipeline,
            net_output: *ActivationTensor,
        ) !void {
            try math.trig.tanh(T, pipeline, net_output);
        }

        pub fn getDerivative(
            _: *const anyopaque,
            pipeline: *Pipeline,
            input: *ActivationTensor,
            derivative: *ActivationTensor,
        ) !void {
            const command_queue = pipeline.command_queue;

            const vectors_enabled = input.flags.vectors_enabled and derivative.flags.vectors_enabled;
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                vectors_enabled,
                .TanhDev,
                "tanh_dev",
                tanh_cl_kernel,
                null,
            );

            const prev_events = pipeline.prevEvents();

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);

            try setArg(kernel, 0, cl_mem_size, @ptrCast(&input.buffer));
            try setArg(kernel, 1, cl_mem_size, @ptrCast(&derivative.buffer));

            var num_elements: u64 = undefined;
            var work_items: u64 = undefined;

            if (vectors_enabled) {
                num_elements = input.memory_layout.number_of_vectors;
                work_items = input.work_configuration.local_work_items_for_vectors_1d[command_queue.wekua_id];
            } else {
                num_elements = input.dimensions.number_of_elements;
                work_items = input.work_configuration.local_work_items_1d[command_queue.wekua_id];
            }

            var new_event: cl.event.Event = undefined;
            try cl.kernel.enqueueNdRange(
                command_queue.cl_command_queue,
                kernel,
                null,
                &.{num_elements},
                &.{work_items},
                prev_events,
                &new_event,
            );
            errdefer helpers.releaseEvent(new_event);

            try pipeline.append(&.{new_event});
        }
    };
}

test {
    std.testing.refAllDecls(Tanh(f32));
    std.testing.refAllDecls(Tanh(f64));
}
