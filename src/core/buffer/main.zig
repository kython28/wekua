
const Context = @import("../context/main.zig");

pub const IOError = error{
    IndexOutOfBounds,
    DifferentContext,
};


pub fn Buffer(comptime T: type) type {
    return struct {
        ctx: *Context,
        ptr: *anyopaque,
        len: usize,

        pub fn init(ctx: *Context, len: usize) Context.AllocError!Self {
            const ptr = try ctx.alloc(len * @sizeOf(T));
            errdefer ctx.free(ptr, len * @sizeOf(T));

            return Self{
                .ctx = ctx,
                .ptr = ptr,
                .len = len,
            };
        }

        pub fn getValue(self: *Self, index: usize) IOError!T {
            if (index >= self.len) {
                return IOError.IndexOutOfBounds;
            }

            const offset = index * @sizeOf(T);
            var value: T = undefined;

            self.ctx.read(self.ptr, offset, &value, @sizeOf(T));
            return value;
        }

        pub fn putValue(self: *Self, index: usize, value: T) IOError!void {
            if (index >= self.len) {
                return IOError.IndexOutOfBounds;
            }

            const offset = index * @sizeOf(T);
            self.ctx.write(self.ptr, offset, &value, @sizeOf(T));
        }

        pub fn readFromHost(self: *Self, offset: usize, host: []const T) IOError!void {
            if (offset + host.len > self.len) {
                return IOError.IndexOutOfBounds;
            }

            const byte_offset = offset * @sizeOf(T);
            self.ctx.write(self.ptr, byte_offset, host.ptr, host.len * @sizeOf(T));
        }

        pub fn writeToHost(self: *Self, offset: usize, host: []T) IOError!void {
            if (offset + host.len > self.len) {
                return IOError.IndexOutOfBounds;
            }

            const byte_offset = offset * @sizeOf(T);
            self.ctx.read(self.ptr, byte_offset, host.ptr, host.len * @sizeOf(T));
        }

        pub fn readFromHostRect(
            self: *Self,
            buffer_origin: [3]usize,
            host_origin: [3]usize,
            region: [3]usize,
            buffer_row_pitch: usize,
            buffer_slice_pitch: usize,
            host_row_pitch: usize,
            host_slice_pitch: usize,
            host: []const T,
        ) IOError!void {
            if (!fitsInBuffer(self.len, buffer_origin, region, buffer_row_pitch, buffer_slice_pitch)) {
                return IOError.IndexOutOfBounds;
            }
            if (!fitsInHostBuffer(host.len, host_origin, region, host_row_pitch, host_slice_pitch)) {
                return IOError.IndexOutOfBounds;
            }

            self.ctx.readRect(
                self.ptr,
                buffer_origin,
                host_origin,
                region,
                buffer_row_pitch * @sizeOf(T),
                buffer_slice_pitch * @sizeOf(T),
                host_row_pitch * @sizeOf(T),
                host_slice_pitch * @sizeOf(T),
                host.ptr,
            );
        }

        pub fn writeToHostRect(
            self: *Self,
            buffer_origin: [3]usize,
            host_origin: [3]usize,
            region: [3]usize,
            buffer_row_pitch: usize,
            buffer_slice_pitch: usize,
            host_row_pitch: usize,
            host_slice_pitch: usize,
            host: []T,
        ) IOError!void {
            if (!fitsInBuffer(self.len, buffer_origin, region, buffer_row_pitch, buffer_slice_pitch)) {
                return IOError.IndexOutOfBounds;
            }
            if (!fitsInHostBuffer(host.len, host_origin, region, host_row_pitch, host_slice_pitch)) {
                return IOError.IndexOutOfBounds;
            }

            self.ctx.writeRect(
                self.ptr,
                buffer_origin,
                host_origin,
                region,
                buffer_row_pitch * @sizeOf(T),
                buffer_slice_pitch * @sizeOf(T),
                host_row_pitch * @sizeOf(T),
                host_slice_pitch * @sizeOf(T),
                host.ptr,
            );
        }

        pub fn copyFrom(
            self: *Self,
            src: *Self,
            src_offset: usize,
            dst_offset: usize,
            len: usize,
        ) IOError!void {
            if (self.ctx != src.ctx) {
                return IOError.DifferentContext;
            }

            if (self.len < dst_offset + len) {
                return IOError.IndexOutOfBounds;
            }
            if (src.len < src_offset + len) {
                return IOError.IndexOutOfBounds;
            }

            self.ctx.copy(src.ptr, self.ptr, src_offset * @sizeOf(T), dst_offset * @sizeOf(T), len * @sizeOf(T));
        }

        pub fn copyFromRect(
            self: *Self,
            src: *Self,
            src_origin: [3]usize,
            dst_origin: [3]usize,
            region: [3]usize,
            src_row_pitch: usize,
            src_slice_pitch: usize,
            dst_row_pitch: usize,
            dst_slice_pitch: usize,
        ) IOError!void {
            if (self.ctx != src.ctx) {
                return IOError.DifferentContext;
            }

            if (!fitsInBuffer(src.len, src_origin, region, src_row_pitch, src_slice_pitch)) {
                return IOError.IndexOutOfBounds;
            }
            if (!fitsInBuffer(self.len, dst_origin, region, dst_row_pitch, dst_slice_pitch)) {
                return IOError.IndexOutOfBounds;
            }

            self.ctx.copyRect(
                src.ptr,
                self.ptr,
                src_origin,
                dst_origin,
                region,
                src_row_pitch * @sizeOf(T),
                src_slice_pitch * @sizeOf(T),
                dst_row_pitch * @sizeOf(T),
                dst_slice_pitch * @sizeOf(T),
            );
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

        pub fn free(self: *Self) void {
            self.ctx.free(self.ptr, self.len * @sizeOf(T));
        }

        const Self = @This();
    };
}
