pub usingnamespace @import("linear.zig");

const wekua = @import("../../wekua.zig");

pub fn Layer(comptime T: type) type {
    const LayerTensor = wekua.Tensor(T);

    return struct {
        pub const VTable = struct {
            deinit: *const fn (ptr: *const anyopaque) void,

            prepareCache: *const fn (
                ptr: *const anyopaque,
                number_of_elements: u64,
            ) anyerror!*anyopaque,

            releaseCache: *const fn (
                ptr: *const anyopaque,
                cache: *anyopaque,
            ) void,

            forward: *const fn (
                ptr: *const anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                input: *LayerTensor,
                cache: ?*anyopaque,
            ) anyerror!*LayerTensor,

            getGradient: *const fn (ptr: *const anyopaque, cache: *anyopaque) *LayerTensor,

            backward: *const fn (
                ptr: *const anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                cache: *anyopaque,
                input_gradient: ?*LayerTensor
            ) anyerror!void,
        };

        ptr: *anyopaque,
        vtable: VTable,

        const Self = @This();

        pub inline fn deinit(self: *Self) void {
            self.vtable.deinit(@ptrCast(self.ptr));
        }

        pub inline fn prepareCache(
            self: *const Self,
            number_of_elements: u64,
        ) !*anyopaque {
            return self.vtable.prepareCache(@ptrCast(self.ptr), number_of_elements);
        }

        pub inline fn releaseCache(
            self: *const Self,
            cache: *anyopaque,
        ) void {
            self.vtable.releaseCache(@ptrCast(self.ptr), cache);
        }

        pub inline fn forward(
            self: *Self,
            command_queue: *const wekua.core.CommandQueue,
            input: *LayerTensor,
            cache: *anyopaque,
        ) !*LayerTensor {
            return self.vtable.forward(@ptrCast(self.ptr), command_queue, input, cache);
        }

        pub inline fn getGradient(
            self: *Self,
            cache: *anyopaque,
        ) *LayerTensor {
            return self.vtable.getGradient(@ptrCast(self.ptr), cache);
        }

        pub inline fn backward(
            self: *Self,
            command_queue: *const wekua.core.CommandQueue,
            cache: *anyopaque,
            input_gradient: ?*LayerTensor,
        ) !void {
            return self.vtable.backward(@ptrCast(self.ptr), command_queue, cache, input_gradient);
        }
    };
}
