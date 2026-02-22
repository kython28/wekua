const std = @import("std");
pub const linear_module = @import("linear.zig");
pub const sequential_module = @import("sequential.zig");

const core = @import("core");
const Pipeline = core.Pipeline;
const file_header = core.file_header;

const tensor_module = @import("tensor");
const TensorErrors = tensor_module.Errors;
const serialization = tensor_module.serialization;

const cache_module = @import("cache.zig");
pub const Cache = cache_module.Cache;

pub fn Layer(comptime T: type) type {
    const Tensor = tensor_module.Tensor(T);

    return struct {
        pub const DumpToFileErrors = serialization.DumpToFileErrors;
        pub const DumpErrors = DumpToFileErrors || std.fs.File.OpenError;
        pub const LoadFromFileErrors = serialization.LoadFromFileErrors;
        pub const LoadErrors = LoadFromFileErrors || std.fs.File.OpenError;

        pub const VTable = struct {
            deinit: *const fn (ptr: *anyopaque, pipeline: *Pipeline) void,

            dumpToFile: *const fn (ptr: *const anyopaque, pipeline: *Pipeline, file: std.fs.File) DumpToFileErrors!void,
            loadFromFile: *const fn (ptr: *const anyopaque, pipeline: *Pipeline, file: std.fs.File) LoadFromFileErrors!void,

            getCachedOutput: *const fn (ptr: *const anyopaque, cache: *const anyopaque) *Tensor,
            getWeights: *const fn (ptr: *const anyopaque) []const *Tensor,
            getBias: *const fn (ptr: *const anyopaque) ?[]const ?*Tensor,

            prepareCache: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                number_of_elements: u64,
            ) TensorErrors!*anyopaque,

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
            ) TensorErrors!*Tensor,

            getSensitivity: *const fn (ptr: *const anyopaque, cache: *const anyopaque) *Tensor,

            backward: *const fn (
                ptr: *const anyopaque,
                pipeline: *Pipeline,
                cache: *anyopaque,
                input: *Tensor,
                input_gradient: ?*Tensor,
            ) TensorErrors!void,

            getGradients: *const fn (ptr: *const anyopaque, cache: *const anyopaque) []const *Tensor,
            getBiasGradients: *const fn (ptr: *const anyopaque, cache: *const anyopaque) ?[]const ?*Tensor,
        };

        ptr: *anyopaque,
        vtable: VTable,

        const Self = @This();

        pub inline fn deinit(self: *const Self, pipeline: *Pipeline) void {
            self.vtable.deinit(@ptrCast(self.ptr), pipeline);
        }

        pub inline fn dumpToFile(self: *const Self, pipeline: *Pipeline, file: std.fs.File) DumpToFileErrors!void {
            return self.vtable.dumpToFile(@ptrCast(self.ptr), pipeline, file);
        }

        pub inline fn loadFromFile(self: *const Self, pipeline: *Pipeline, file: std.fs.File) LoadFromFileErrors!void {
            return self.vtable.loadFromFile(@ptrCast(self.ptr), pipeline, file);
        }

        pub fn dump(self: *const Self, pipeline: *Pipeline, path: []const u8) DumpErrors!void {
            const file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            try self.dumpToFile(pipeline, file);
        }

        pub fn load(self: *const Self, pipeline: *Pipeline, path: []const u8) LoadErrors!void {
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            try self.loadFromFile(pipeline, file);
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
        ) TensorErrors!*anyopaque {
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
        ) TensorErrors!*Tensor {
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
        ) TensorErrors!void {
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
    _ = linear_module;
    _ = sequential_module;
    _ = cache_module;

    std.testing.refAllDecls(Layer(f32));
    std.testing.refAllDecls(Layer(f64));
}
