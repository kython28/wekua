const std = @import("std");

const CommandQueue = @import("../command_queue.zig");
const Buffer = @import("../buffer.zig");

pub const AllocError = std.mem.Allocator.Error;

pub const VTable = struct {
    alloc: *const fn (
        ctx_ptr: *anyopaque,
        len: usize,
    ) ?*anyopaque,

    free: *const fn (
        ctx_ptr: *anyopaque,
        buf: *anyopaque,
    ) void,

    createCommandQueue: *const fn (ctx_ptr: *anyopaque) AllocError!CommandQueue,

    deinit: *const fn (ctx_ptr: *anyopaque) void,
};

ptr: *anyopaque,
vtable: VTable,

pub inline fn alloc(self: *Context, comptime T: type, len: usize) AllocError!*Buffer(T) {
    const ptr = self.vtable.alloc(self.ptr, len) orelse return AllocError.OutOfMemory;
    return @ptrCast(ptr);
}

pub inline fn free(self: *Context, comptime T: type, ptr: *Buffer(T)) void {
    self.vtable.free(self.ptr, @ptrCast(ptr));
}

pub inline fn createCommandQueue(self: *Context) AllocError!CommandQueue {
    return self.vtable.createCommandQueue(self.ptr);
}

const Context = @This();
