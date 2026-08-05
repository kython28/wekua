const std = @import("std");

const Context = @import("context/main.zig");


/// Parameters for a buffer->host read. All sizes and offsets are in
/// bytes, not elements. The element-counting wrapper in `Buffer(T)`
/// converts before constructing this payload.
pub const ReadCommand = struct {
    /// Opaque handle of the source buffer. Must remain alive until the
    /// queue reports the command done.
    buf: *anyopaque,
    /// Byte offset inside `buf` at which to start reading. Must be a
    /// multiple of the backend's alignment.
    offset: usize,
    /// Destination host buffer. The number of bytes to read is inferred
    /// from `dst.len`. Must remain alive until the queue reports the
    /// command done.
    dst: []u8,
};

/// Parameters for a host->buffer write. All sizes and offsets are in
/// bytes, not elements. The element-counting wrapper in `Buffer(T)`
/// converts before constructing this payload.
pub const WriteCommand = struct {
    /// Opaque handle of the destination buffer. Must remain alive until
    /// the queue reports the command done.
    buf: *anyopaque,
    /// Byte offset inside `buf` at which to start writing. Must be a
    /// multiple of the backend's alignment.
    offset: usize,
    /// Source host buffer. The number of bytes to write is inferred from
    /// `src.len`. Must remain alive until the queue reports the command
    /// done.
    src: []const u8,
};

/// Parameters for a 3D rectangular buffer->host read command.
///
/// All sizes and offsets are in bytes, not elements. The
/// element-counting wrapper in `Buffer(T)` converts before constructing
/// this payload.
pub const ReadRectCommand = struct {
    /// Opaque handle of the source buffer. Must remain alive until the
    /// queue reports the command done.
    buf: *anyopaque,
    /// `(x, y, z)` byte offset of the region's lower corner inside the
    /// buffer. Must be a multiple of the backend's alignment.
    buffer_origin: [3]usize,
    /// `(x, y, z)` byte offset of the region's lower corner inside
    /// `dst`. Same alignment as `buffer_origin`.
    host_origin: [3]usize,
    /// `(width, height, depth)` of the region, in bytes. Each component
    /// must be non-zero.
    region: [3]usize,
    /// Stride in bytes between two consecutive rows in the buffer.
    /// Must satisfy `buffer_row_pitch >= region[0]`. Larger values
    /// leave padding between rows.
    buffer_row_pitch: usize,
    /// Stride in bytes between two consecutive 2D slices in the
    /// buffer. Must satisfy
    /// `buffer_slice_pitch >= region[0] * region[1]`. Larger values
    /// leave padding between slices.
    buffer_slice_pitch: usize,
    /// Stride in bytes between two consecutive rows in `dst`. Same
    /// constraints as `buffer_row_pitch`.
    host_row_pitch: usize,
    /// Stride in bytes between two consecutive 2D slices in `dst`.
    /// Same constraints as `buffer_slice_pitch`.
    host_slice_pitch: usize,
    /// Destination host buffer. Must remain alive until the queue
    /// reports the command done.
    dst: []u8,
};

/// Parameters for a 3D rectangular host->buffer write command.
///
/// All sizes and offsets are in bytes, not elements. The
/// element-counting wrapper in `Buffer(T)` converts before constructing
/// this payload.
pub const WriteRectCommand = struct {
    /// Opaque handle of the destination buffer. Must remain alive until
    /// the queue reports the command done.
    buf: *anyopaque,
    /// `(x, y, z)` byte offset of the region's lower corner inside the
    /// buffer. Must be a multiple of the backend's alignment.
    buffer_origin: [3]usize,
    /// `(x, y, z)` byte offset of the region's lower corner inside
    /// `src`. Same alignment as `buffer_origin`.
    host_origin: [3]usize,
    /// `(width, height, depth)` of the region, in bytes. Each component
    /// must be non-zero.
    region: [3]usize,
    /// Stride in bytes between two consecutive rows in the buffer.
    /// Must satisfy `buffer_row_pitch >= region[0]`. Larger values
    /// leave padding between rows.
    buffer_row_pitch: usize,
    /// Stride in bytes between two consecutive 2D slices in the
    /// buffer. Must satisfy
    /// `buffer_slice_pitch >= region[0] * region[1]`. Larger values
    /// leave padding between slices.
    buffer_slice_pitch: usize,
    /// Stride in bytes between two consecutive rows in `src`. Same
    /// constraints as `buffer_row_pitch`.
    host_row_pitch: usize,
    /// Stride in bytes between two consecutive 2D slices in `src`.
    /// Same constraints as `buffer_slice_pitch`.
    host_slice_pitch: usize,
    /// Source host buffer. Must remain alive until the queue reports
    /// the command done.
    src: []const u8,
};

/// Parameters for a buffer-to-buffer copy. All sizes and offsets are
/// in bytes, not elements. The element-counting wrapper in `Buffer(T)`
/// converts before constructing this payload.
pub const CopyCommand = struct {
    /// Opaque handle of the source buffer. Must remain alive until the
    /// queue reports the command done.
    src_buf: *anyopaque,
    /// Opaque handle of the destination buffer. Must remain alive until
    /// the queue reports the command done.
    dst_buf: *anyopaque,
    /// Byte offset inside `src_buf` at which to start reading. Must be
    /// a multiple of the backend's alignment. `src_offset + len` must
    /// not overflow the source buffer.
    src_offset: usize,
    /// Byte offset inside `dst_buf` at which to start writing. Must be
    /// a multiple of the backend's alignment. `dst_offset + len` must
    /// not overflow the destination buffer.
    dst_offset: usize,
    /// Number of bytes to copy. Source and destination ranges must not
    /// overlap.
    len: usize,
};

/// Parameters for a 3D rectangular buffer-to-buffer copy. All sizes,
/// origins, and pitches are in bytes, not elements. The
/// element-counting wrapper in `Buffer(T)` converts before constructing
/// this payload.
pub const CopyRectCommand = struct {
    /// Opaque handle of the source buffer. Must remain alive until the
    /// queue reports the command done.
    src_buf: *anyopaque,
    /// Opaque handle of the destination buffer. Must remain alive until
    /// the queue reports the command done. Source and destination
    /// ranges must not overlap.
    dst_buf: *anyopaque,
    /// `(x, y, z)` byte offset of the region's lower corner inside
    /// `src_buf`. Must be a multiple of the backend's alignment.
    src_origin: [3]usize,
    /// `(x, y, z)` byte offset of the region's lower corner inside
    /// `dst_buf`. Same alignment as `src_origin`.
    dst_origin: [3]usize,
    /// `(width, height, depth)` of the region, in bytes. Each component
    /// must be non-zero.
    region: [3]usize,
    /// Stride in bytes between two consecutive rows in `src_buf`. Must
    /// satisfy `src_row_pitch >= region[0]`. Larger values leave
    /// padding between rows.
    src_row_pitch: usize,
    /// Stride in bytes between two consecutive 2D slices in `src_buf`.
    /// Must satisfy `src_slice_pitch >= region[0] * region[1]`. Larger
    /// values leave padding between slices.
    src_slice_pitch: usize,
    /// Stride in bytes between two consecutive rows in `dst_buf`. Same
    /// constraints as `src_row_pitch`.
    dst_row_pitch: usize,
    /// Stride in bytes between two consecutive 2D slices in `dst_buf`.
    /// Same constraints as `src_slice_pitch`.
    dst_slice_pitch: usize,
};

/// Parameters for a buffer fill. All sizes and offsets are in bytes,
/// not elements. The element-counting wrapper in `Buffer(T)` converts
/// before constructing this payload.
pub const FillCommand = struct {
    /// Opaque handle of the destination buffer. Must remain alive until
    /// the queue reports the command done.
    buf: *anyopaque,
    /// Byte offset inside `buf` at which to start filling. Must be a
    /// multiple of the backend's alignment.
    offset: usize,
    /// Number of bytes to fill. The filled range is
    /// `[offset, offset + len)`. Must be a whole multiple of
    /// `pattern.len`.
    len: usize,
    /// Tile pattern, in bytes. The pattern is repeated end-to-end
    /// across the filled range. Must remain alive until the queue
    /// reports the command done.
    pattern: []const u8,
};

/// Parameters for a 3D rectangular buffer fill. Sizes, origin, and
/// pitches are in bytes; the operation is buffer-local (no host-side
/// payload). The element-counting wrapper in `Buffer(T)` converts
/// before constructing this payload.
pub const FillRectCommand = struct {
    /// Opaque handle of the destination buffer. Must remain alive until
    /// the queue reports the command done.
    buf: *anyopaque,
    /// `(x, y, z)` byte offset of the region's lower corner inside the
    /// buffer. Must be a multiple of the backend's alignment.
    buffer_origin: [3]usize,
    /// `(width, height, depth)` of the region, in bytes. Each component
    /// must be non-zero. The total fill extent
    /// `region[0] * region[1] * region[2]` must be a whole multiple of
    /// `pattern.len`.
    region: [3]usize,
    /// Stride in bytes between two consecutive rows in the buffer.
    /// Must satisfy `buffer_row_pitch >= region[0]`. Larger values
    /// leave padding between rows.
    buffer_row_pitch: usize,
    /// Stride in bytes between two consecutive 2D slices in the
    /// buffer. Must satisfy
    /// `buffer_slice_pitch >= region[0] * region[1]`. Larger values
    /// leave padding between slices.
    buffer_slice_pitch: usize,
    /// Tile pattern, in bytes. The pattern is repeated end-to-end
    /// across the filled region. Must remain alive until the queue
    /// reports the command done.
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
