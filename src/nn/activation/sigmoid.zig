const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const helpers = tensor_module.helpers;

const activation = @import("main.zig");

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

        pub fn run(_: *const anyopaque, pipeline: *Pipeline, net_output: *ActivationTensor) !void {
            const command_queue = pipeline.command_queue;

            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                net_output.flags.vectors_enabled,
                .Sigmoid,
                "sigmoid",
                sigmoid_cl_kernel,
                null,
            );

            const prev_events = pipeline.prevEvents();

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);

            try setArg(kernel, 0, cl_mem_size, @ptrCast(&net_output.buffer));

            var num_elements: u64 = undefined;
            var work_items: u64 = undefined;

            if (net_output.flags.vectors_enabled) {
                num_elements = net_output.memory_layout.number_of_vectors;
                work_items = net_output.work_configuration.local_work_items_for_vectors_1d[command_queue.wekua_id];
            }else{
                num_elements = net_output.dimensions.number_of_elements;
                work_items = net_output.work_configuration.local_work_items_1d[command_queue.wekua_id];
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

        pub fn getDerivative(
            _: *const anyopaque,
            pipeline: *Pipeline,
            output: *ActivationTensor,
            derivative: *ActivationTensor,
        ) !void {
            const command_queue = pipeline.command_queue;

            const vectors_enabled = output.flags.vectors_enabled and derivative.flags.vectors_enabled;
            const kernel = try KernelsSet.getClKernel(
                T,
                command_queue,
                vectors_enabled,
                .SigmoidDev,
                "sigmoid_dev",
                sigmoid_cl_kernel,
                null,
            );

            const prev_events = pipeline.prevEvents();

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);

            try setArg(kernel, 0, cl_mem_size, @ptrCast(&output.buffer));
            try setArg(kernel, 1, cl_mem_size, @ptrCast(&derivative.buffer));

            var num_elements: u64 = undefined;
            var work_items: u64 = undefined;

            if (vectors_enabled) {
                num_elements = output.memory_layout.number_of_vectors;
                work_items = output.work_configuration.local_work_items_for_vectors_1d[command_queue.wekua_id];
            } else {
                num_elements = output.dimensions.number_of_elements;
                work_items = output.work_configuration.local_work_items_1d[command_queue.wekua_id];
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
    std.testing.refAllDecls(Sigmoid(f32));
    std.testing.refAllDecls(Sigmoid(f64));
}
