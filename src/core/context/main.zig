const std = @import("std");

const CommandQueue = @import("../command_queue.zig");
const Buffer = @import("../buffer.zig");

/// Alias for `std.mem.Allocator.Error`. Backend wrappers that allocate surface
/// failures through this set.
pub const AllocError = std.mem.Allocator.Error;

/// Backend dispatch table for a `Context`. Backends populate every field; the
/// public `Context` API forwards through this table.
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

/// Allocate a `Buffer(T)` of `len` elements.
///
/// Returns `OutOfMemory` if the backend cannot satisfy the request. The
/// returned buffer is uninitialized.
pub inline fn alloc(self: *Context, comptime T: type, len: usize) AllocError!*Buffer(T) {
    const ptr = self.vtable.alloc(self.ptr, len) orelse return AllocError.OutOfMemory;
    return @ptrCast(ptr);
}

/// Release a buffer previously returned by `alloc`.
///
/// The caller is responsible for draining any in-flight work on the buffer
/// (typically by calling `cq.wait()` on the queue that owns it) before
/// freeing.
pub inline fn free(self: *Context, comptime T: type, ptr: *Buffer(T)) void {
    self.vtable.free(self.ptr, @ptrCast(ptr));
}

/// Create a new command queue for this context.
///
/// Returns `OutOfMemory` if the backend cannot allocate its bookkeeping. The
/// queue must be released with `CommandQueue.deinit` when no longer needed.
pub inline fn createCommandQueue(self: *Context) AllocError!CommandQueue {
    return self.vtable.createCommandQueue(self.ptr);
}

const Context = @This();
