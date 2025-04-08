pub usingnamespace @import("linear.zig");

const wekua = @import("../../wekua.zig");
const w_cache = @import("cache.zig");

pub fn Layer(comptime T: type) type {
    const LayerTensor = wekua.Tensor(T);

    return struct {
        pub const Cache = w_cache.Cache(T);

        pub const VTable = struct {
            deinit: *const fn (ptr: *const anyopaque) void,

            getWeights: *const fn (ptr: *const anyopaque) []const *LayerTensor,
            getBias: *const fn (ptr: *const anyopaque) ?[]const *LayerTensor,

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
                cache: *anyopaque,
            ) anyerror!*LayerTensor,

            getSensitivity: *const fn (ptr: *const anyopaque, cache: *anyopaque) *LayerTensor,

            backward: *const fn (
                ptr: *const anyopaque,
                command_queue: *const wekua.core.CommandQueue,
                cache: *anyopaque,
                input: *LayerTensor,
                input_gradient: ?*LayerTensor
            ) anyerror!void,

            getGradients: *const fn (ptr: *const anyopaque, cache: *anyopaque) []const *LayerTensor,
            getBiasGradients: *const fn (ptr: *const anyopaque, cache: *anyopaque) ?[]const *LayerTensor,
        };

        ptr: *anyopaque,
        vtable: VTable,

        const Self = @This();

        pub inline fn deinit(self: *const Self) void {
            self.vtable.deinit(@ptrCast(self.ptr));
        }

        pub inline fn getWeights(self: *const Self) []const *LayerTensor {
            return self.vtable.getWeights(@ptrCast(self.ptr));
        }

        pub inline fn getBias(self: *const Self) ?[]const *LayerTensor {
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
            input: *LayerTensor,
            cache: *anyopaque,
        ) !*LayerTensor {
            return self.vtable.forward(@ptrCast(self.ptr), command_queue, input, cache);
        }

        pub inline fn getSensitivity(
            self: *const Self,
            cache: *anyopaque,
        ) *LayerTensor {
            return self.vtable.getSensitivity(@ptrCast(self.ptr), cache);
        }

        pub inline fn backward(
            self: *const Self,
            command_queue: *const wekua.core.CommandQueue,
            cache: *anyopaque,
            input: *LayerTensor,
            input_gradient: ?*LayerTensor,
        ) !void {
            return self.vtable.backward(@ptrCast(self.ptr), command_queue, cache, input, input_gradient);
        }

        pub inline fn getGradients(
            self: *const Self,
            cache: *anyopaque,
        ) []const *LayerTensor {
            return self.vtable.getGradients(@ptrCast(self.ptr), cache);
        }

        pub inline fn getBiasGradients(
            self: *const Self,
            cache: *anyopaque,
        ) ?[]const *LayerTensor {
            return self.vtable.getBiasGradients(@ptrCast(self.ptr), cache);
        }
    };
}
