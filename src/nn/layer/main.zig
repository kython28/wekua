pub usingnamespace @import("linear.zig");
pub usingnamespace @import("sequential.zig");

const wekua = @import("../../wekua.zig");
pub usingnamespace @import("cache.zig");


pub fn Layer(comptime T: type) type {
    const Tensor = wekua.Tensor(T);

    return struct {
        pub const VTable = struct {
            deinit: *const fn (ptr: *const anyopaque) void,

            getCachedOutput: *const fn (ptr: *const anyopaque, cache: *const anyopaque) *Tensor,
            getWeights: *const fn (ptr: *const anyopaque) []const *Tensor,
            getBias: *const fn (ptr: *const anyopaque) ?[]const ?*Tensor,

            prepareCache: *const fn (
                ptr: *const anyopaque,
                number_of_elements: u64,
            ) anyerror!*anyopaque,

            releaseCache: *const fn (
                ptr: *const anyopaque,
                cache: *const anyopaque,
            ) void,

            forward: *const fn (
                ptr: *const anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                input: *Tensor,
                cache: *anyopaque,
            ) anyerror!*Tensor,

            getSensitivity: *const fn (ptr: *const anyopaque, cache: *const anyopaque) *Tensor,

            backward: *const fn (
                ptr: *const anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                cache: *anyopaque,
                input: *Tensor,
                input_gradient: ?*Tensor
            ) anyerror!void,

            getGradients: *const fn (ptr: *const anyopaque, cache: *const anyopaque) []const *Tensor,
            getBiasGradients: *const fn (ptr: *const anyopaque, cache: *const anyopaque) ?[]const ?*Tensor,
        };

        ptr: *anyopaque,
        vtable: VTable,

        const Self = @This();

        pub inline fn deinit(self: *const Self) void {
            self.vtable.deinit(@ptrCast(self.ptr));
        }

        pub inline fn getCachedOutput(self: *const Self, cache: *anyopaque) *Tensor {
            return self.vtable.getCachedOutput(@ptrCast(self.ptr), cache);
        }

        pub inline fn getWeights(self: *const Self) []const *Tensor {
            return self.vtable.getWeights(@ptrCast(self.ptr));
        }

        pub inline fn getBias(self: *const Self) ?[]const ?*Tensor {
            return self.vtable.getBias(@ptrCast(self.ptr));
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
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            input: *Tensor,
            cache: *anyopaque,
        ) !*Tensor {
            return self.vtable.forward(@ptrCast(self.ptr), command_queue, input, cache);
        }

        pub inline fn getSensitivity(
            self: *const Self,
            cache: *anyopaque,
        ) *Tensor {
            return self.vtable.getSensitivity(@ptrCast(self.ptr), cache);
        }

        pub inline fn backward(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            cache: *anyopaque,
            input: *Tensor,
            input_gradient: ?*Tensor,
        ) !void {
            return self.vtable.backward(@ptrCast(self.ptr), command_queue, cache, input, input_gradient);
        }

        pub inline fn getGradients(
            self: *const Self,
            cache: *anyopaque,
        ) []const *Tensor {
            return self.vtable.getGradients(@ptrCast(self.ptr), cache);
        }

        pub inline fn getBiasGradients(
            self: *const Self,
            cache: *anyopaque,
        ) ?[]const ?*Tensor {
            return self.vtable.getBiasGradients(@ptrCast(self.ptr), cache);
        }
    };
}
