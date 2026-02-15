const builtin = @import("builtin");
const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const CommandQueue = core.CommandQueue;

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;

fn unmap_tensor_buffer(
    comptime T: type,
    command_queue: *const CommandQueue,
    buffer: cl.buffer.Mem,
    map: []T,
) !void {
    var unmap_event: cl.event.Event = undefined;
    try cl.buffer.unmap([]T, command_queue.cl_command_queue, buffer, map, null, &unmap_event);

    try cl.event.wait(unmap_event);
}

inline fn printPadding(writer: anytype, padding: usize) !void {
    for (0..padding) |_| try writer.writeByte(' ');
}

inline fn openBracket(writer: anytype, padding: usize) !void {
    try printPadding(writer, padding);
    try writer.writeByte('[');
}

inline fn closeBracket(writer: anytype, padding: usize) !void {
    try printPadding(writer, padding);
    try writer.writeAll("],");
}

inline fn printComplexIntegerValue(
    comptime T: type,
    comptime max_value_str_len: []const u8,
    writer: anytype,
    index: usize,
    tmp_buffer: []u8,
    buf: []const T,
) !void {
    var formatted_buf: []u8 = undefined;
    const value = buf[index];
    const real_value = value.real;
    const imag_value = value.imag;

    const real_is_zero = (real_value == 0);
    const imag_is_zero = (imag_value == 0);

    if (!real_is_zero and !imag_is_zero) {
        if (imag_value < 0) {
            formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{d}{d}j", .{ real_value, imag_value });
        } else {
            formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{d}+{d}j", .{ real_value, imag_value });
        }
    } else if (!real_is_zero and imag_is_zero) {
        formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{d}", .{real_value});
    } else if (real_is_zero and !imag_is_zero) {
        formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{d}j", .{imag_value});
    } else {
        formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{d}", .{real_value});
    }

    try writer.print("{s: >" ++ max_value_str_len ++ "},", .{formatted_buf});
}

inline fn printComplexFloatValue(
    comptime T: type,
    writer: anytype,
    index: usize,
    tmp_buffer: []u8,
    buf: []const T,
) !void {
    var formatted_buf: []u8 = undefined;
    const value = buf[index];
    const real_value = value.real;
    const imag_value = value.imag;

    const real_is_zero = (@abs(real_value) < 1e-8);
    const imag_is_zero = (@abs(imag_value) < 1e-8);

    if (!real_is_zero and !imag_is_zero) {
        if (imag_value < 0) {
            formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{e:.2}{e:.2}j", .{ real_value, imag_value });
        } else {
            formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{e:.2}+{e:.2}j", .{ real_value, imag_value });
        }
    } else if (!real_is_zero and imag_is_zero) {
        formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{e:.5}", .{real_value});
    } else if (real_is_zero and !imag_is_zero) {
        formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{e:.5}j", .{imag_value});
    } else {
        formatted_buf = try std.fmt.bufPrint(tmp_buffer, "{e:.5}", .{real_value});
    }

    try writer.print("{s: >24},", .{formatted_buf});
}

fn printVector(
    comptime T: type,
    writer: anytype,
    padding: usize,
    buf: []const T,
    cols: u64,
) !void {
    try printPadding(writer, padding);
    const is_complex = comptime core.types.isComplex(T);
    const SubType = core.types.getType(T);
    switch (@typeInfo(SubType)) {
        .int => |int_info| {
            const max_value = switch (int_info.signedness) {
                .signed => comptime std.math.minInt(SubType),
                .unsigned => comptime std.math.maxInt(SubType),
            };
            const max_value_str = switch (is_complex) {
                true => switch (int_info.signedness) {
                    .signed => comptime std.fmt.comptimePrint("{d}{d}j", .{ max_value, max_value }),
                    .unsigned => comptime std.fmt.comptimePrint("{d}+{d}j", .{ max_value, max_value }),
                },
                false => comptime std.fmt.comptimePrint("{d}", .{max_value}),
            };
            const max_value_str_len = comptime std.fmt.comptimePrint("{d}", .{max_value_str.len + 1});

            comptime var max_items: usize = @max(if (is_complex) 3 else 5, (65 / max_value_str.len));
            if ((max_items % 2) == 0) max_items += 1;
            const width = @min(cols, (max_items - 1) / 2);

            switch (is_complex) {
                true => {
                    var tmp_buffer: [max_value_str.len + 1]u8 = undefined;
                    for (0..width) |i| {
                        try printComplexIntegerValue(T, max_value_str_len, writer, i, &tmp_buffer, buf);
                    }

                    if (cols > width) {
                        var dec: usize = width + @as(usize, 1);
                        if (cols > max_items) {
                            dec = width;
                            try writer.print("{s: >" ++ max_value_str_len ++ "},", .{"..."});
                        }
                        for (@max(width, cols - dec)..cols) |i| {
                            try printComplexIntegerValue(T, max_value_str_len, writer, i, &tmp_buffer, buf);
                        }
                    }
                },
                false => {
                    for (buf[0..width]) |v| {
                        try writer.print("{d: >" ++ max_value_str_len ++ "},", .{v});
                    }

                    if (cols > width) {
                        var dec: usize = width + @as(usize, 1);
                        if (cols > max_items) {
                            dec = width;
                            try writer.print("{s: >" ++ max_value_str_len ++ "},", .{"..."});
                        }
                        for (buf[@max(width, cols - dec)..cols]) |v| {
                            try writer.print("{d: >" ++ max_value_str_len ++ "},", .{v});
                        }
                    }
                },
            }
        },
        .float => {
            switch (is_complex) {
                true => {
                    var tmp_buffer: [24]u8 = undefined;
                    for (0..@min(cols, 2)) |i| {
                        try printComplexFloatValue(T, writer, i, &tmp_buffer, buf);
                    }

                    if (cols > 2) {
                        var dec: usize = 3;
                        if (cols > 5) {
                            dec = 2;
                            try writer.print("{s: >24},", .{"..."});
                        }
                        for (@max(2, cols - dec)..cols) |i| {
                            try printComplexFloatValue(T, writer, i, &tmp_buffer, buf);
                        }
                    }
                },
                false => {
                    const limit = 4;
                    for (buf[0..@min(cols, limit)]) |v| {
                        if ((@abs(10000.0) - @abs(v)) > 1e-8) {
                            try writer.print("{d: >14.8},", .{v});
                        } else {
                            try writer.print("{e: >14.5},", .{v});
                        }
                    }

                    if (cols > limit) {
                        var dec: usize = limit + 1;
                        if (cols > (limit * 2 + 1)) {
                            dec = limit;
                            try writer.print("{s: >14},", .{"..."});
                        }

                        for (buf[@max(limit, cols - dec)..cols]) |v| {
                            if ((@abs(10000.0) - @abs(v)) > 1e-8) {
                                try writer.print("{d: >14.8},", .{v});
                            } else {
                                try writer.print("{e: >14.5},", .{v});
                            }
                        }
                    }
                },
            }
        },
        else => @compileError("Type not supported"),
    }
}

inline fn printVectorOrMatrix(
    comptime T: type,
    writer: anytype,
    padding: usize,
    buf: []const T,
    pitches: []const u64,
    shape: []const u64,
) !void {
    if (pitches.len == 1) {
        try printVector(T, writer, padding, buf[0..shape[0]], shape[0]);
    } else {
        const pitch = pitches[0];
        const rows = shape[0];
        const cols = shape[1];
        for (0..@min(2, rows)) |i| {
            try writer.writeByte('\n');
            try printVector(
                T,
                writer,
                padding + 2,
                buf[(i * pitch)..((i + 1) * pitch)],
                cols,
            );
        }

        if (rows > 2) {
            if (rows > 4) {
                try writer.writeByte('\n');
                try printPadding(writer, padding + 2);
                try writer.writeAll(" ... ");
            }

            for (@max(2, rows - 2)..rows) |i| {
                try writer.writeByte('\n');
                try printVector(
                    T,
                    writer,
                    padding + 2,
                    buf[(i * pitch)..((i + 1) * pitch)],
                    cols,
                );
            }
        }
        try writer.writeByte('\n');
    }
}

fn printDim(
    comptime T: type,
    writer: anytype,
    padding: usize,
    buf: []T,
    shape: []const u64,
    pitches: []const u64,
) !void {
    try openBracket(writer, padding);
    if (pitches.len > 2) {
        try writer.writeByte('\n');
        const dim_size = shape[0];
        const pitch = pitches[0];
        for (0..@min(2, dim_size)) |i| {
            try printDim(
                T,
                writer,
                padding + 2,
                buf[(i * pitch)..((i + 1) * pitch)],
                shape[1..],
                pitches[1..],
            );
        }

        if (dim_size > 2) {
            if (dim_size > 4) {
                try writer.writeByte('\n');
                try printPadding(writer, padding + 2);
                try writer.writeAll("... \n");
            }

            for (@max(2, (dim_size - 2))..dim_size) |i| {
                try printDim(
                    T,
                    writer,
                    padding + 2,
                    buf[(i * pitch)..((i + 1) * pitch)],
                    shape[1..],
                    pitches[1..],
                );
            }
        }
        try writer.writeByte('\n');
    } else {
        try printVectorOrMatrix(T, writer, padding, buf, pitches, shape);
    }
    try closeBracket(writer, padding);
    if (padding > 0) try writer.writeByte('\n');
}

pub fn printZ(
    comptime T: type,
    pipeline: *Pipeline,
    writer: anytype,
    tensor: *Tensor(T),
) !void {
    const command_queue = pipeline.command_queue;
    const prev_events = pipeline.prevEvents();

    var mapping_event: cl.event.Event = undefined;
    const memory_map = try cl.buffer.map(
        []T,
        command_queue.cl_command_queue,
        tensor.buffer,
        false,
        cl.buffer.MapFlag.read,
        0,
        tensor.memory_layout.size,
        prev_events,
        &mapping_event,
    );
    try cl.event.wait(mapping_event);
    defer unmap_tensor_buffer(T, command_queue, tensor.buffer, memory_map) catch |err| {
        std.debug.panic("Error unmapping tensor buffer: {s}\n", .{@errorName(err)});
    };

    const pitches = tensor.dimensions.pitches;
    const shape = tensor.dimensions.shape;
    try writer.writeAll("Tensor(");

    const is_complex = comptime core.types.isComplex(T);
    try printDim(T, writer, 0, memory_map, shape, pitches);

    try writer.print(" shape=({d}", .{shape[0]});
    for (shape[1..]) |s| {
        try writer.print(", {d}", .{s});
    }
    try writer.writeByte(')');

    if (is_complex) {
        const child_type = core.types.getType(T);
        try writer.print(", dtype=complex_{s})\n", .{@typeName(child_type)});
    } else {
        try writer.print(", dtype={s})\n", .{@typeName(T)});
    }
}

pub fn print(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) !void {
    const allocator = pipeline.command_queue.context.allocator;

    // TODO: Update to new Writer
    var array: std.ArrayList(u8) = .empty;
    defer array.deinit(allocator);

    const array_writer = array.writer(allocator);
    try printZ(T, pipeline, &array_writer, tensor);
    if (builtin.is_test) {
        std.log.warn("{s}", .{array.items});
    }else{
        var stdout_file = std.fs.File.stdout();
        try stdout_file.writeAll(array.items);
    }
}
