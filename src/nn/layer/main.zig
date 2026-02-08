pub const linear = @import("linear.zig");
pub const sequential = @import("sequential.zig");

const core = @import("core");
const Pipeline = core.Pipeline;

const tensor_module = @import("tensor");
pub const Cache = @import("cache.zig").Cache;

pub fn Layer(comptime T: type) type {
    const Tensor = tensor_module.Tensor(T);

    return struct {
        pub const VTable = struct {
            deinit: *const fn (ptr: *const anyopaque, pipeline: *Pipeline) void,

            getCachedOutput: *const fn (ptr: *const anyopaque, cache: *const anyopaque) *Tensor,
            getWeights: *const fn (ptr: *const anyopaque) []const *Tensor,
            getBias: *const fn (ptr: *const anyopaque) ?[]const ?*Tensor,

            prepareCache: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                number_of_elements: u64,
            ) anyerror!*anyopaque,

            releaseCache: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                cache: *const anyopaque,
            ) void,

            forward: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                input: *Tensor,
                cache: *anyopaque,
            ) anyerror!*Tensor,

            getSensitivity: *const fn (ptr: *const anyopaque, cache: *const anyopaque) *Tensor,

            backward: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                cache: *anyopaque,
                input: *Tensor,
                input_gradient: ?*Tensor,
            ) anyerror!void,

            getGradients: *const fn (ptr: *const anyopaque, cache: *const anyopaque) []const *Tensor,
            getBiasGradients: *const fn (ptr: *const anyopaque, cache: *const anyopaque) ?[]const ?*Tensor,
        };

        ptr: *anyopaque,
        vtable: VTable,

        const Self = @This();

        pub inline fn deinit(self: *const Self, pipeline: *Pipeline) void {
            self.vtable.deinit(@ptrCast(self.ptr), pipeline);
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
            pipeline: *Pipeline,
            number_of_elements: u64,
        ) !*anyopaque {
            return self.vtable.prepareCache(@ptrCast(self.ptr), pipeline, number_of_elements);
        }

        pub inline fn releaseCache(
            self: *const Self,
            pipeline: *Pipeline,
            cache: *anyopaque,
        ) void {
            self.vtable.releaseCache(@ptrCast(self.ptr), pipeline, cache);
        }

        pub inline fn forward(
            self: *const Self,
            pipeline: *Pipeline,
            input: *Tensor,
            cache: *anyopaque,
        ) !*Tensor {
            return self.vtable.forward(@ptrCast(self.ptr), pipeline, input, cache);
        }

        pub inline fn getSensitivity(
            self: *const Self,
            cache: *anyopaque,
        ) *Tensor {
            return self.vtable.getSensitivity(@ptrCast(self.ptr), cache);
        }

        pub inline fn backward(
            self: *const Self,
            pipeline: *Pipeline,
            cache: *anyopaque,
            input: *Tensor,
            input_gradient: ?*Tensor,
        ) !void {
            return self.vtable.backward(@ptrCast(self.ptr), pipeline, cache, input, input_gradient);
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

test {
    _ = linear;
    _ = sequential;
    _ = @import("cache.zig");
}
