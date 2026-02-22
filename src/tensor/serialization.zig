const std = @import("std");
const core = @import("core");
const Pipeline = core.Pipeline;

const file_header = core.file_header;
const WekuaFileHeader = file_header.WekuaFileHeader;

pub const TensorMetadata = extern struct {
    type_index: u8,
    ndim: u8,
    reserved: u16 = 0,
};

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const memory = @import("memory/main.zig");

pub const SerializationErrors = error{
    TypeMismatch,
    ShapeMismatch,
};

pub const DumpToFileErrors = TensorErrors || SerializationErrors || std.fs.File.WriteError;
pub const DumpErrors = DumpToFileErrors || std.fs.File.OpenError;
pub const LoadFromFileErrors = TensorErrors || SerializationErrors || std.fs.File.ReadError || file_header.ValidationError;
pub const LoadErrors = LoadFromFileErrors || std.fs.File.OpenError;

fn byteSwapBuffer(comptime T: type, buffer: []T) void {
    const native_endian = comptime @import("builtin").cpu.arch.endian();
    if (native_endian == .little) return;

    if (comptime core.types.isComplex(T)) {
        for (buffer) |*elem| {
            elem.real = @byteSwap(elem.real);
            elem.imag = @byteSwap(elem.imag);
        }
    } else {
        for (buffer) |*elem| {
            elem.* = @byteSwap(elem.*);
        }
    }
}

pub fn dumpToFile(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    file: std.fs.File,
) DumpToFileErrors!void {
    const element_size = @sizeOf(T);
    const shape = tensor.dimensions.shape;
    const ndim = shape.len;
    const num_elements = tensor.dimensions.number_of_elements_without_padding;

    const allocator = pipeline.allocator;

    // TODO: use mmap on the file region instead of a temp buffer
    const buffer = try allocator.alloc(T, num_elements);
    defer allocator.free(buffer);

    try memory.writeToBuffer(T, pipeline, tensor, buffer);
    pipeline.waitAndCleanup();

    byteSwapBuffer(T, buffer);

    const header = WekuaFileHeader{};
    try file.writeAll(std.mem.asBytes(&header));

    const meta = TensorMetadata{
        .type_index = @intCast(core.types.getTypeIndex(T)),
        .ndim = @intCast(ndim),
    };
    try file.writeAll(std.mem.asBytes(&meta));

    for (shape) |dim| {
        const le_dim = std.mem.nativeToLittle(u64, dim);
        try file.writeAll(std.mem.asBytes(&le_dim));
    }

    const data_bytes: []const u8 = @as([*]const u8, @ptrCast(buffer.ptr))[0 .. num_elements * element_size];
    try file.writeAll(data_bytes);
}

pub fn dump(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    path: []const u8,
) DumpErrors!void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    try dumpToFile(T, pipeline, tensor, file);
}

pub fn loadFromFile(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    file: std.fs.File,
) LoadFromFileErrors!void {
    const element_size = @sizeOf(T);
    const shape = tensor.dimensions.shape;
    const ndim = shape.len;
    const num_elements = tensor.dimensions.number_of_elements_without_padding;

    var header: WekuaFileHeader = undefined;
    const header_bytes = std.mem.asBytes(&header);
    const header_read = try file.readAll(header_bytes);
    if (header_read != header_bytes.len) return error.InvalidMagic;
    try file_header.validate(header, .tensor);

    var meta: TensorMetadata = undefined;
    const meta_bytes = std.mem.asBytes(&meta);
    const meta_read = try file.readAll(meta_bytes);
    if (meta_read != meta_bytes.len) return error.TypeMismatch;

    const expected_type_index: u8 = @intCast(core.types.getTypeIndex(T));
    if (meta.type_index != expected_type_index) return error.TypeMismatch;

    if (meta.ndim != @as(u8, @intCast(ndim))) return error.ShapeMismatch;

    for (shape) |expected_dim| {
        var le_dim: u64 = undefined;
        const dim_bytes = std.mem.asBytes(&le_dim);
        const dim_read = try file.readAll(dim_bytes);
        if (dim_read != dim_bytes.len) return error.ShapeMismatch;

        const stored_dim = std.mem.littleToNative(u64, le_dim);
        if (stored_dim != expected_dim) return error.ShapeMismatch;
    }

    const allocator = pipeline.allocator;

    // TODO: use mmap on the file region instead of a temp buffer
    const buffer = try allocator.alloc(T, num_elements);
    defer allocator.free(buffer);

    const data_bytes: []u8 = @as([*]u8, @ptrCast(buffer.ptr))[0 .. num_elements * element_size];
    const data_read = try file.readAll(data_bytes);
    if (data_read != data_bytes.len) return error.ShapeMismatch;

    byteSwapBuffer(T, buffer);

    try memory.readFromBuffer(T, pipeline, tensor, buffer);
    pipeline.waitAndCleanup();
}

pub fn load(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    path: []const u8,
) LoadErrors!void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    try loadFromFile(T, pipeline, tensor, file);
}

const testing = std.testing;
const cl = @import("opencl");
const Context = core.Context;

test "serialization round-trip via path - all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};
    const tmp_path = "/tmp/wekua_test_serialization.wkt";

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const num_elements = shape[0] * shape[1];

            const src = try Tensor(T).empty(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const input_buffer = try allocator.alloc(T, num_elements);
            defer allocator.free(input_buffer);

            for (input_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i + 1), .imag = @floatFromInt(i + 100) },
                        .int => .{ .real = @intCast(i + 1), .imag = @intCast(i + 100) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i + 1),
                        .int => @intCast(i + 1),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, src, input_buffer);
            pipeline.waitAndCleanup();

            try dump(T, pipeline, src, tmp_path);

            const dst = try Tensor(T).empty(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            try load(T, pipeline, dst, tmp_path);

            const output_buffer = try allocator.alloc(T, num_elements);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            for (input_buffer, output_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }

            std.fs.cwd().deleteFile(tmp_path) catch {};
        }
    }
}

test "serialization round-trip via file handle - all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 5 };
    const config = tensor_module.CreateConfig{};
    const tmp_path = "/tmp/wekua_test_serialization_fh.wkt";

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const num_elements = shape[0] * shape[1];

            const src = try Tensor(T).empty(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const input_buffer = try allocator.alloc(T, num_elements);
            defer allocator.free(input_buffer);

            for (input_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 3), .imag = @floatFromInt(i * 7) },
                        .int => .{ .real = @intCast(i * 3), .imag = @intCast(i * 7) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 3),
                        .int => @intCast(i * 3),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, src, input_buffer);
            pipeline.waitAndCleanup();

            {
                const file = try std.fs.cwd().createFile(tmp_path, .{});
                defer file.close();
                try dumpToFile(T, pipeline, src, file);
            }

            const dst = try Tensor(T).empty(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            {
                const file = try std.fs.cwd().openFile(tmp_path, .{});
                defer file.close();
                try loadFromFile(T, pipeline, dst, file);
            }

            const output_buffer = try allocator.alloc(T, num_elements);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            for (input_buffer, output_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }

            std.fs.cwd().deleteFile(tmp_path) catch {};
        }
    }
}

test "serialization - invalid magic" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const tmp_path = "/tmp/wekua_test_invalid_magic.wkt";

    {
        const file = try std.fs.cwd().createFile(tmp_path, .{});
        defer file.close();
        try file.writeAll(&[_]u8{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 });
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    const shape = [_]u64{ 2, 3 };
    const tensor = try Tensor(f32).empty(context, pipeline, &shape, .{});
    defer tensor.release(pipeline);

    const result = load(f32, pipeline, tensor, tmp_path);
    try testing.expectError(error.InvalidMagic, result);
}

test "serialization - type mismatch" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const tmp_path = "/tmp/wekua_test_type_mismatch.wkt";

    const src = try Tensor(f32).empty(context, pipeline, &shape, .{});
    defer src.release(pipeline);
    try dump(f32, pipeline, src, tmp_path);
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    const dst = try Tensor(i32).empty(context, pipeline, &shape, .{});
    defer dst.release(pipeline);

    const result = load(i32, pipeline, dst, tmp_path);
    try testing.expectError(error.TypeMismatch, result);
}

test "serialization - shape mismatch" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const tmp_path = "/tmp/wekua_test_shape_mismatch.wkt";

    const shape1 = [_]u64{ 3, 4 };
    const src = try Tensor(f32).empty(context, pipeline, &shape1, .{});
    defer src.release(pipeline);
    try dump(f32, pipeline, src, tmp_path);
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    const shape2 = [_]u64{ 2, 5 };
    const dst = try Tensor(f32).empty(context, pipeline, &shape2, .{});
    defer dst.release(pipeline);

    const result = load(f32, pipeline, dst, tmp_path);
    try testing.expectError(error.ShapeMismatch, result);
}

test "serialization - multi-dimensional 1D, 2D, 3D" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const config = tensor_module.CreateConfig{};
    const tmp_path = "/tmp/wekua_test_multidim.wkt";

    {
        const shape = [_]u64{10};
        const num_elements: usize = 10;

        const src = try Tensor(f32).empty(context, pipeline, &shape, config);
        defer src.release(pipeline);

        const input_buffer = try allocator.alloc(f32, num_elements);
        defer allocator.free(input_buffer);
        for (input_buffer, 0..) |*val, i| val.* = @floatFromInt(i + 1);

        try memory.readFromBuffer(f32, pipeline, src, input_buffer);
        pipeline.waitAndCleanup();

        try dump(f32, pipeline, src, tmp_path);

        const dst = try Tensor(f32).empty(context, pipeline, &shape, config);
        defer dst.release(pipeline);
        try load(f32, pipeline, dst, tmp_path);

        const output_buffer = try allocator.alloc(f32, num_elements);
        defer allocator.free(output_buffer);
        try memory.writeToBuffer(f32, pipeline, dst, output_buffer);
        pipeline.waitAndCleanup();

        for (input_buffer, output_buffer) |expected, result_val| {
            try testing.expectEqual(expected, result_val);
        }

        std.fs.cwd().deleteFile(tmp_path) catch {};
    }

    {
        const shape = [_]u64{ 5, 7 };
        const num_elements: usize = 5 * 7;

        const src = try Tensor(f32).empty(context, pipeline, &shape, config);
        defer src.release(pipeline);

        const input_buffer = try allocator.alloc(f32, num_elements);
        defer allocator.free(input_buffer);
        for (input_buffer, 0..) |*val, i| val.* = @floatFromInt(i + 1);

        try memory.readFromBuffer(f32, pipeline, src, input_buffer);
        pipeline.waitAndCleanup();

        try dump(f32, pipeline, src, tmp_path);

        const dst = try Tensor(f32).empty(context, pipeline, &shape, config);
        defer dst.release(pipeline);
        try load(f32, pipeline, dst, tmp_path);

        const output_buffer = try allocator.alloc(f32, num_elements);
        defer allocator.free(output_buffer);
        try memory.writeToBuffer(f32, pipeline, dst, output_buffer);
        pipeline.waitAndCleanup();

        for (input_buffer, output_buffer) |expected, result_val| {
            try testing.expectEqual(expected, result_val);
        }

        std.fs.cwd().deleteFile(tmp_path) catch {};
    }

    {
        const shape = [_]u64{ 2, 3, 4 };
        const num_elements: usize = 2 * 3 * 4;

        const src = try Tensor(f32).empty(context, pipeline, &shape, config);
        defer src.release(pipeline);

        const input_buffer = try allocator.alloc(f32, num_elements);
        defer allocator.free(input_buffer);
        for (input_buffer, 0..) |*val, i| val.* = @floatFromInt(i + 1);

        try memory.readFromBuffer(f32, pipeline, src, input_buffer);
        pipeline.waitAndCleanup();

        try dump(f32, pipeline, src, tmp_path);

        const dst = try Tensor(f32).empty(context, pipeline, &shape, config);
        defer dst.release(pipeline);
        try load(f32, pipeline, dst, tmp_path);

        const output_buffer = try allocator.alloc(f32, num_elements);
        defer allocator.free(output_buffer);
        try memory.writeToBuffer(f32, pipeline, dst, output_buffer);
        pipeline.waitAndCleanup();

        for (input_buffer, output_buffer) |expected, result_val| {
            try testing.expectEqual(expected, result_val);
        }

        std.fs.cwd().deleteFile(tmp_path) catch {};
    }
}

test {
    std.testing.refAllDecls(@This());
}
