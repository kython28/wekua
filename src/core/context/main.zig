const std = @import("std");

const CommandQueue = @import("../command_queue.zig");

pub const AllocError = std.mem.Allocator.Error;

pub const VTable = struct {
    alloc: *const fn (
        ctx_ptr: *anyopaque,
        len: usize,
    ) ?*anyopaque,

    free: *const fn (
        ctx_ptr: *anyopaque,
        buf: *anyopaque,
        len: usize,
    ) void,

    createCommandQueue: *const fn (ctx_ptr: *anyopaque) CommandQueue,
};

ptr: *anyopaque,
vtable: VTable,

pub inline fn alloc(self: *Context, len: usize) AllocError!*anyopaque {
    return self.vtable.alloc(self.ptr, len) orelse return AllocError.OutOfMemory;
}

pub inline fn free(self: *Context, ptr: *anyopaque, len: usize) void {
    self.vtable.free(self.ptr, ptr, len);
}

pub inline fn createCommandQueue(self: *Context) CommandQueue {
    return self.vtable.createCommandQueue(self.ptr);
}

const Context = @This();
