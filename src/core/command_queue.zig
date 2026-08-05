const std = @import("std");

const Context = @import("context/main.zig");


pub const ReadCommand = struct {
    buf: *anyopaque,
    offset: usize,
    dst: []u8,
};

pub const WriteCommand = struct {
    buf: *anyopaque,
    offset: usize,
    src: []const u8,
};

pub const ReadRectCommand = struct {
    buf: *anyopaque,
    buffer_origin: [3]usize,
    host_origin: [3]usize,
    region: [3]usize,
    buffer_row_pitch: usize,
    buffer_slice_pitch: usize,
    host_row_pitch: usize,
    host_slice_pitch: usize,
    dst: []u8,
};

pub const WriteRectCommand = struct {
    buf: *anyopaque,
    buffer_origin: [3]usize,
    host_origin: [3]usize,
    region: [3]usize,
    buffer_row_pitch: usize,
    buffer_slice_pitch: usize,
    host_row_pitch: usize,
    host_slice_pitch: usize,
    src: []const u8,
};

pub const CopyCommand = struct {
    src_buf: *anyopaque,
    dst_buf: *anyopaque,
    src_offset: usize,
    dst_offset: usize,
    len: usize,
};

pub const CopyRectCommand = struct {
    src_buf: *anyopaque,
    dst_buf: *anyopaque,
    src_origin: [3]usize,
    dst_origin: [3]usize,
    region: [3]usize,
    src_row_pitch: usize,
    src_slice_pitch: usize,
    dst_row_pitch: usize,
    dst_slice_pitch: usize,
};

pub const FillCommand = struct {
    buf: *anyopaque,
    offset: usize,
    len: usize,
    pattern: []const u8,
};

pub const FillRectCommand = struct {
    buf: *anyopaque,
    buffer_origin: [3]usize,
    region: [3]usize,
    buffer_row_pitch: usize,
    buffer_slice_pitch: usize,
    pattern: []const u8,
};

pub const CommandTag = enum {
    read,
    write,

    read_rect,
    write_rect,

    copy,
    copy_rect,

    fill,
    fill_rect,
};

pub const Command = union(CommandTag) {
    read: ReadCommand,
    write: WriteCommand,

    read_rect: ReadRectCommand,
    write_rect: WriteRectCommand,

    copy: CopyCommand,
    copy_rect: CopyRectCommand,

    fill: FillCommand,
    fill_rect: FillRectCommand,
};

pub const Error = error{
    OutOfMemory,
    InvalidBuffer,
    OutOfBounds,
    InvalidPitch,
} || std.Io.Cancelable;

pub const VTable = struct {
    deinit: *const fn (*anyopaque) void,
    enqueue: *const fn (*anyopaque, Command) Error!void,
    wait: *const fn (*anyopaque) std.Io.Cancelable!void,
};

ptr: *anyopaque,
vtable: VTable,

pub inline fn enqueue(self: *CommandQueue, command: Command) Error!void {
    return self.vtable.enqueue(self.ptr, command);
}

pub inline fn deinit(self: *CommandQueue) void {
    self.vtable.deinit(self.ptr);
}

pub inline fn wait(self: *CommandQueue) std.Io.Cancelable!void {
    self.vtable.wait(self.ptr);
}


const CommandQueue = @This();
