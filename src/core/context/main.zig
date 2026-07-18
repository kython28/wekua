const std = @import("std");

const Buffer = @import("../buffer/main.zig");

pub const AllocError = std.mem.Allocator.AllocError;

const LinearBufToHostIO = *const fn (
    ctx_ptr: *anyopaque,
    buf: *anyopaque,
    offset: usize,
    ptr: *anyopaque,
    len: usize,
) void;

const RectBufToHostIO = *const fn (
    ctx_ptr: *anyopaque,
    buf: *anyopaque,
    buffer_origin: [3]usize,
    host_origin: [3]usize,
    region: [3]usize,
    buffer_row_pitch: usize,
    buffer_slice_pitch: usize,
    host_row_pitch: usize,
    host_slice_pitch: usize,
    ptr: *anyopaque,
) void;

pub const VTable = struct {
    alloc: *const fn (*anyopaque, usize, std.mem.Alignment) ?*anyopaque,
    free: *const fn (*anyopaque, *anyopaque, usize) void,

    read: LinearBufToHostIO,
    write: LinearBufToHostIO,

    readRect: RectBufToHostIO,
    writeRect: RectBufToHostIO,

    copy: *const fn (*anyopaque, *anyopaque, *anyopaque, usize, usize, usize) void,
    copyRect: *const fn (
        ctx_ptr: *anyopaque,
        src_buf: *anyopaque,
        dst_buf: *anyopaque,
        src_origin: [3]usize,
        dst_origin: [3]usize,
        region: [3]usize,
        src_row_pitch: usize,
        src_slice_pitch: usize,
        dst_row_pitch: usize,
        dst_slice_pitch: usize,
    ) void,
};

ptr: *anyopaque,
vtable: VTable,

pub inline fn alloc(ctx: *Context, len: usize) AllocError!*anyopaque {
    return ctx.vtable.alloc(ctx.ptr, len) orelse return AllocError.OutOfMemory;
}

pub inline fn free(ctx: *Context, ptr: *anyopaque, len: usize) void {
    ctx.vtable.free(ctx.ptr, ptr, len);
}

pub inline fn read(
    ctx: *Context,
    buf: *anyopaque,
    offset: usize,
    ptr: *anyopaque,
    len: usize,
) void {
    ctx.vtable.read(ctx.ptr, buf, offset, ptr, len);
}

pub inline fn write(
    ctx: *Context,
    buf: *anyopaque,
    offset: usize,
    ptr: *anyopaque,
    len: usize,
) void {
    ctx.vtable.write(ctx.ptr, buf, offset, ptr, len);
}

pub inline fn readRect(
    ctx: *Context,
    buf: *anyopaque,
    buffer_origin: [3]usize,
    host_origin: [3]usize,
    region: [3]usize,
    buffer_row_pitch: usize,
    buffer_slice_pitch: usize,
    host_row_pitch: usize,
    host_slice_pitch: usize,
    ptr: *anyopaque,
) void {
    ctx.vtable.readRect(
        ctx.ptr,
        buf,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
    );
}

pub inline fn writeRect(
    ctx: *Context,
    buf: *anyopaque,
    buffer_origin: [3]usize,
    host_origin: [3]usize,
    region: [3]usize,
    buffer_row_pitch: usize,
    buffer_slice_pitch: usize,
    host_row_pitch: usize,
    host_slice_pitch: usize,
    ptr: *anyopaque,
) void {
    ctx.vtable.writeRect(
        ctx.ptr,
        buf,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
    );
}

pub inline fn copy(
    ctx: *Context,
    src_buf: *anyopaque,
    dst_buf: *anyopaque,
    src_offset: usize,
    dst_offset: usize,
    len: usize,
) void {
    ctx.vtable.copy(ctx.ptr, src_buf, dst_buf, src_offset, dst_offset, len);
}

pub inline fn copyRect(
    ctx: *Context,
    src_buf: *anyopaque,
    dst_buf: *anyopaque,
    src_origin: [3]usize,
    dst_origin: [3]usize,
    region: [3]usize,
    src_row_pitch: usize,
    src_slice_pitch: usize,
    dst_row_pitch: usize,
    dst_slice_pitch: usize,
) void {
    ctx.vtable.copyRect(
        ctx.ptr,
        src_buf,
        dst_buf,
        src_origin,
        dst_origin,
        region,
        src_row_pitch,
        src_slice_pitch,
        dst_row_pitch,
        dst_slice_pitch,
    );
}

const Context = @This();
