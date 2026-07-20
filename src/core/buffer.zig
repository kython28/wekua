const std = @import("std");

const Context = @import("context/main.zig");
const CommandQueue = @import("command_queue.zig");

pub const Error = error{
    IndexOutOfBounds,
    DifferentContext,
    InvalidPattern,
} || CommandQueue.Error;

pub fn Buffer(comptime T: type) type {
    return struct {
        ptr: *anyopaque,
        len: usize,

        pub fn init(ctx: *Context, len: usize) Context.AllocError!Self {
            const ptr = try ctx.alloc(len * @sizeOf(T));
            errdefer ctx.free(ptr, len * @sizeOf(T));

            return Self{
                .ptr = ptr,
                .len = len,
            };
        }

        pub fn deinit(self: *Self, ctx: *Context, cq: *CommandQueue) void {
            cq.wait();
            ctx.free(self.ptr, self.len * @sizeOf(T));
        }

        pub fn getValue(self: *Self, cq: *CommandQueue, index: usize) Error!T {
            if (index >= self.len) {
                return Error.IndexOutOfBounds;
            }

            var value: T = undefined;
            try cq.enqueue(.{
                .read = .{
                    .buf = self.ptr,
                    .offset = index * @sizeOf(T),
                    .ptr = &value,
                    .len = @sizeOf(T),
                },
            });
            cq.wait();

            return value;
        }

        pub fn putValue(self: *Self, cq: *CommandQueue, index: usize, value: T) Error!void {
            if (index >= self.len) {
                return Error.IndexOutOfBounds;
            }

            try cq.enqueue(.{
                .write = .{
                    .buf = self.ptr,
                    .offset = index * @sizeOf(T),
                    .ptr = &value,
                    .len = @sizeOf(T),
                },
            });
            cq.wait();
        }

        pub fn readFromHost(self: *Self, cq: *CommandQueue, offset: usize, host: []const T) Error!void {
            if (offset + host.len > self.len) {
                return Error.IndexOutOfBounds;
            }

            try cq.enqueue(.{
                .write = .{
                    .buf = self.ptr,
                    .offset = offset * @sizeOf(T),
                    .ptr = host.ptr,
                    .len = host.len * @sizeOf(T),
                },
            });
            cq.wait();
        }

        pub fn writeToHost(self: *Self, cq: *CommandQueue, offset: usize, host: []T) Error!void {
            if (offset + host.len > self.len) {
                return Error.IndexOutOfBounds;
            }

            try cq.enqueue(.{
                .read = .{
                    .buf = self.ptr,
                    .offset = offset * @sizeOf(T),
                    .ptr = host.ptr,
                    .len = host.len * @sizeOf(T),
                },
            });
            cq.wait();
        }

        pub fn readFromHostRect(
            self: *Self,
            cq: *CommandQueue,
            buffer_origin: [3]usize,
            host_origin: [3]usize,
            region: [3]usize,
            buffer_row_pitch: usize,
            buffer_slice_pitch: usize,
            host_row_pitch: usize,
            host_slice_pitch: usize,
            host: []const T,
        ) Error!void {
            if (!fitsInBuffer(self.len, buffer_origin, region, buffer_row_pitch, buffer_slice_pitch)) {
                return Error.IndexOutOfBounds;
            }
            if (!fitsInHostBuffer(host.len, host_origin, region, host_row_pitch, host_slice_pitch)) {
                return Error.IndexOutOfBounds;
            }

            try cq.enqueue(.{
                .write_rect = .{
                    .buf = self.ptr,
                    .buffer_origin = buffer_origin,
                    .host_origin = host_origin,
                    .region = region,
                    .buffer_row_pitch = buffer_row_pitch * @sizeOf(T),
                    .buffer_slice_pitch = buffer_slice_pitch * @sizeOf(T),
                    .host_row_pitch = host_row_pitch * @sizeOf(T),
                    .host_slice_pitch = host_slice_pitch * @sizeOf(T),
                    .ptr = host.ptr,
                },
            });
            cq.wait();
        }

        pub fn writeToHostRect(
            self: *Self,
            cq: *CommandQueue,
            buffer_origin: [3]usize,
            host_origin: [3]usize,
            region: [3]usize,
            buffer_row_pitch: usize,
            buffer_slice_pitch: usize,
            host_row_pitch: usize,
            host_slice_pitch: usize,
            host: []T,
        ) Error!void {
            if (!fitsInBuffer(self.len, buffer_origin, region, buffer_row_pitch, buffer_slice_pitch)) {
                return Error.IndexOutOfBounds;
            }
            if (!fitsInHostBuffer(host.len, host_origin, region, host_row_pitch, host_slice_pitch)) {
                return Error.IndexOutOfBounds;
            }

            try cq.enqueue(.{
                .read_rect = .{
                    .buf = self.ptr,
                    .buffer_origin = buffer_origin,
                    .host_origin = host_origin,
                    .region = region,
                    .buffer_row_pitch = buffer_row_pitch * @sizeOf(T),
                    .buffer_slice_pitch = buffer_slice_pitch * @sizeOf(T),
                    .host_row_pitch = host_row_pitch * @sizeOf(T),
                    .host_slice_pitch = host_slice_pitch * @sizeOf(T),
                    .ptr = host.ptr,
                },
            });
            cq.wait();
        }

        pub fn fill(self: *Self, cq: *CommandQueue, offset: usize, len: usize, pattern: []const T) Error!void {
            if (offset + len > self.len) {
                return Error.IndexOutOfBounds;
            }

            const pattern_bytes = std.mem.sliceAsBytes(pattern);
            if (!isValidPattern(pattern_bytes, len * @sizeOf(T))) {
                return Error.InvalidPattern;
            }

            try cq.enqueue(.{
                .fill = .{
                    .buf = self.ptr,
                    .offset = offset * @sizeOf(T),
                    .len = len * @sizeOf(T),
                    .pattern = pattern_bytes,
                },
            });
            cq.wait();
        }

        pub fn fillRect(
            self: *Self,
            cq: *CommandQueue,
            buffer_origin: [3]usize,
            region: [3]usize,
            buffer_row_pitch: usize,
            buffer_slice_pitch: usize,
            pattern: []const T,
        ) Error!void {
            if (!fitsInBuffer(self.len, buffer_origin, region, buffer_row_pitch, buffer_slice_pitch)) {
                return Error.IndexOutOfBounds;
            }

            const pattern_bytes = std.mem.sliceAsBytes(pattern);
            const region_len = region[0] * region[1] * region[2];
            if (!isValidPattern(pattern_bytes, region_len * @sizeOf(T))) {
                return Error.InvalidPattern;
            }

            try cq.enqueue(.{
                .fill_rect = .{
                    .buf = self.ptr,
                    .buffer_origin = buffer_origin,
                    .region = region,
                    .buffer_row_pitch = buffer_row_pitch * @sizeOf(T),
                    .buffer_slice_pitch = buffer_slice_pitch * @sizeOf(T),
                    .pattern = pattern_bytes,
                },
            });
            cq.wait();
        }

        fn isValidPattern(pattern_bytes: []const u8, byte_len: usize) bool {
            if (pattern_bytes.len == 0) return false;
            if (byte_len % pattern_bytes.len != 0) return false;

            return true;
        }

        pub fn copyFrom(
            self: *Self,
            src: *Self,
            cq: *CommandQueue,
            src_offset: usize,
            dst_offset: usize,
            len: usize,
        ) Error!void {
            if (self.len < dst_offset + len) {
                return Error.IndexOutOfBounds;
            }
            if (src.len < src_offset + len) {
                return Error.IndexOutOfBounds;
            }

            try cq.enqueue(.{
                .copy = .{
                    .src_buf = src.ptr,
                    .dst_buf = self.ptr,
                    .src_offset = src_offset * @sizeOf(T),
                    .dst_offset = dst_offset * @sizeOf(T),
                    .len = len * @sizeOf(T),
                },
            });
        }

        pub fn copyFromRect(
            self: *Self,
            src: *Self,
            cq: *CommandQueue,
            src_origin: [3]usize,
            dst_origin: [3]usize,
            region: [3]usize,
            src_row_pitch: usize,
            src_slice_pitch: usize,
            dst_row_pitch: usize,
            dst_slice_pitch: usize,
        ) Error!void {
            if (!fitsInBuffer(src.len, src_origin, region, src_row_pitch, src_slice_pitch)) {
                return Error.IndexOutOfBounds;
            }
            if (!fitsInBuffer(self.len, dst_origin, region, dst_row_pitch, dst_slice_pitch)) {
                return Error.IndexOutOfBounds;
            }

            try cq.enqueue(.{
                .copy_rect = .{
                    .src_buf = src.ptr,
                    .dst_buf = self.ptr,
                    .src_origin = src_origin,
                    .dst_origin = dst_origin,
                    .region = region,
                    .src_row_pitch = src_row_pitch * @sizeOf(T),
                    .src_slice_pitch = src_slice_pitch * @sizeOf(T),
                    .dst_row_pitch = dst_row_pitch * @sizeOf(T),
                    .dst_slice_pitch = dst_slice_pitch * @sizeOf(T),
                },
            });
        }

        fn fitsInBuffer(
            len: usize,
            origin: [3]usize,
            region: [3]usize,
            row_pitch: usize,
            slice_pitch: usize,
        ) bool {
            if (row_pitch == 0 or slice_pitch == 0) return false;
            if (slice_pitch % row_pitch != 0) return false;

            if (origin[0] + region[0] > row_pitch) return false;
            if (origin[1] + region[1] > slice_pitch / row_pitch) return false;
            if (origin[2] + region[2] > len / slice_pitch) return false;

            return true;
        }

        fn fitsInHostBuffer(
            len: usize,
            origin: [3]usize,
            region: [3]usize,
            row_pitch: usize,
            slice_pitch: usize,
        ) bool {
            if (row_pitch == 0 or slice_pitch == 0) return false;
            if (slice_pitch % row_pitch != 0) return false;

            if (origin[0] + region[0] > row_pitch) return false;
            if (origin[1] + region[1] > slice_pitch / row_pitch) return false;
            if (origin[2] + region[2] > len / slice_pitch) return false;

            return true;
        }

        const Self = @This();
    };
}
