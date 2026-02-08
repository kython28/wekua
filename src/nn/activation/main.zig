pub const Sigmoid = @import("sigmoid.zig").Sigmoid;
pub const Tanh = @import("tanh.zig").Tanh;

const core = @import("core");
const Pipeline = core.Pipeline;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;

// TODO: Implement activations for integers

pub fn Activation(comptime T: type) type {
    const ActivationTensor = Tensor(T);

    return struct {
        pub const VTable = struct {
            run: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                net_output: *ActivationTensor,
            ) anyerror!void,

            getDerivative: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                input: *ActivationTensor,
                derivative: *ActivationTensor,
            ) anyerror!void,
        };

        ptr: *anyopaque,
        vtable: VTable,

        const Self = @This();

        pub inline fn run(
            self: *const Self,
            pipeline: *Pipeline,
            net_output: *ActivationTensor,
        ) !void {
            try self.vtable.run(@ptrCast(self.ptr), pipeline, net_output);
        }

        pub inline fn getDerivative(
            self: *const Self,
            pipeline: *Pipeline,
            output: *ActivationTensor,
            derivative: *ActivationTensor,
        ) !void {
            try tensor_module.helpers.eqlTensors(T, output, derivative);
            try self.vtable.getDerivative(@ptrCast(self.ptr), pipeline, output, derivative);
        }
    };
}

test {
    _ = @import("sigmoid.zig");
    _ = @import("tanh.zig");
}
