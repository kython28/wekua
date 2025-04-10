pub usingnamespace @import("sigmoid.zig");
pub usingnamespace @import("tanh.zig");

const wekua = @import("../../wekua.zig");
const Tensor = wekua.Tensor;

// TODO: Implement activations for integers

pub fn Activation(comptime T: type) type {
    const ActivationTensor = Tensor(T);

    return struct {
        pub const VTable = struct {
            run: *const fn (
                ptr: *const anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                net_output: *ActivationTensor,
            ) anyerror!void,

            getDerivative: *const fn (
                ptr: *const anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                input: *ActivationTensor,
                derivative: *ActivationTensor,
            ) anyerror!void,
        };

        ptr: *anyopaque,
        vtable: VTable,

        const Self = @This();

        pub inline fn run(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            net_output: *ActivationTensor,
        ) !void {
            try self.vtable.run(@ptrCast(self.ptr), command_queue, net_output);
        }

        pub inline fn getDerivative(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            output: *ActivationTensor,
            derivative: *ActivationTensor,
        ) !void {
            try wekua.tensor.helpers.eqlTensors(T, output, derivative);
            try self.vtable.getDerivative(@ptrCast(self.ptr), command_queue, output, derivative);
        }
    };
}
