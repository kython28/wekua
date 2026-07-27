const std = @import("std");

const Context = @import("context/main.zig");
const CommandQueue = @import("command_queue.zig");

pub fn Buffer(comptime T: type) type {
    return opaque {
        /// Releases the buffer and frees its backing memory.
        ///
        /// `cq.wait()` is invoked first so every in-flight operation queued
        /// against `self` on `cq` completes before the memory is returned to
        /// `ctx`. After this call `self` is invalid.
        ///
        /// `ctx` - Context that allocated this buffer.
        /// `cq`  - Command queue whose pending work must drain before freeing.
        pub fn deinit(self: *Self, ctx: *Context, cq: *CommandQueue) void {
            cq.wait();
            ctx.free(self);
        }

        /// Reads a single element at `index` from the buffer.
        ///
        /// The read is enqueued on `cq` and the queue is waited on before
        /// returning, so the returned `T` is ready to use.
        ///
        /// `cq`    - Command queue used to schedule the read.
        /// `index` - Element index (in elements, not bytes) to read from.
        ///
        /// Returns the element stored at `index`.
        pub fn getValue(self: *Self, cq: *CommandQueue, index: usize) CommandQueue.Error!T {
            var value: T = undefined;
            try cq.enqueue(.{
                .read = .{
                    .buf = @ptrCast(self),
                    .offset = index * @sizeOf(T),
                    .dst = std.mem.asBytes(&value),
                },
            });
            cq.wait();

            return value;
        }

        /// Writes a single `value` at `index` into the buffer.
        ///
        /// The write is enqueued on `cq` and the queue is waited on before
        /// returning, so the value is persisted when this call returns.
        ///
        /// `cq`    - Command queue used to schedule the write.
        /// `index` - Element index (in elements, not bytes) to write to.
        /// `value` - Value to store at `index`.
        pub fn putValue(self: *Self, cq: *CommandQueue, index: usize, value: T) CommandQueue.Error!void {
            try cq.enqueue(.{
                .write = .{
                    .buf = @ptrCast(self),
                    .offset = index * @sizeOf(T),
                    .src = std.mem.asBytes(&value),
                },
            });
            cq.wait();
        }

        /// Copies `host` into the buffer starting at element `offset`.
        ///
        /// `host.len` elements of type `T` are written. The operation is
        /// enqueued on `cq` and returns immediately without waiting.
        ///
        /// Depending on the backend, the contents of `host` may or may not be
        /// copied immediately, so the caller is responsible for calling
        /// `cq.wait()` and keeping `host` alive for the whole lifetime of the
        /// execution to avoid any issue.
        ///
        /// `cq`     - Command queue used to schedule the write.
        /// `offset` - Element index (in elements, not bytes) of the buffer where
        ///            the copy starts.
        /// `host`   - Source slice; `host.len` is the number of elements to copy.
        pub fn readFromHost(self: *Self, cq: *CommandQueue, offset: usize, host: []const T) CommandQueue.Error!void {
            try cq.enqueue(.{
                .write = .{
                    .buf = @ptrCast(self),
                    .offset = offset * @sizeOf(T),
                    .src = std.mem.sliceAsBytes(host),
                },
            });
        }

        /// Copies a contiguous range of the buffer into `host`.
        ///
        /// `host.len` elements of type `T` are read starting at element
        /// `offset` of the buffer. The operation is enqueued on `cq` and
        /// returns immediately without waiting.
        ///
        /// Depending on the backend, the contents of `host` may or may not be
        /// filled immediately, so the caller is responsible for calling
        /// `cq.wait()` and keeping `host` alive for the whole lifetime of the
        /// execution to avoid any issue.
        ///
        /// `cq`     - Command queue used to schedule the read.
        /// `offset` - Element index (in elements, not bytes) of the buffer where
        ///            the read starts.
        /// `host`   - Destination slice; `host.len` is the number of elements
        ///            to read.
        pub fn writeToHost(self: *Self, cq: *CommandQueue, offset: usize, host: []T) CommandQueue.Error!void {
            try cq.enqueue(.{
                .read = .{
                    .buf = @ptrCast(self),
                    .offset = offset * @sizeOf(T),
                    .dst = std.mem.sliceAsBytes(host),
                },
            });
        }

        /// Copies a 3D rectangular region of `host` into the buffer.
        ///
        /// The region is described in elements. `region` is `(width, height,
        /// depth)`: how many elements to copy along each axis. `host_origin`
        /// and `buffer_origin` are `(x, y, z)` offsets giving the lower corner
        /// of the region inside `host` and the buffer respectively.
        ///
        /// Pitches are the stride, in elements, between consecutive rows and
        /// consecutive 2D slices of a 3D layout:
        /// - `host_row_pitch`   - stride between row `y` and `y + 1` in `host`.
        /// - `host_slice_pitch` - stride between slice `z` and `z + 1` in `host`.
        /// - `buffer_row_pitch`   - stride between row `y` and `y + 1` in the buffer.
        /// - `buffer_slice_pitch` - stride between slice `z` and `z + 1` in the
        ///   buffer.
        ///
        /// Each pitch must be at least `region[0]` (row) or `region[0] *
        /// region[1]` (slice); larger values leave padding between rows or
        /// slices. Pitches are expressed in elements and converted to bytes
        /// internally. The operation is enqueued on `cq` and returns
        /// immediately without waiting.
        ///
        /// Depending on the backend, the contents of `host` may or may not be
        /// copied immediately, so the caller is responsible for calling
        /// `cq.wait()` and keeping `host` alive for the whole lifetime of the
        /// execution to avoid any issue.
        ///
        /// `cq`                - Command queue used to schedule the write.
        /// `buffer_origin`     - `(x, y, z)` offset of the region's lower corner
        ///                       inside the buffer, in elements.
        /// `host_origin`       - `(x, y, z)` offset of the region's lower corner
        ///                       inside `host`, in elements.
        /// `region`            - `(width, height, depth)` of the region, in
        ///                       elements.
        /// `buffer_row_pitch`  - Row stride inside the buffer, in elements.
        /// `buffer_slice_pitch`- Slice stride inside the buffer, in elements.
        /// `host_row_pitch`   - Row stride inside `host`, in elements.
        /// `host_slice_pitch`  - Slice stride inside `host`, in elements.
        /// `host`              - Source slice backing the host-side 3D layout.
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
        ) CommandQueue.Error!void {
            try cq.enqueue(.{
                .write_rect = .{
                    .buf = @ptrCast(self),
                    .buffer_origin = buffer_origin,
                    .host_origin = host_origin,
                    .region = region,
                    .buffer_row_pitch = buffer_row_pitch * @sizeOf(T),
                    .buffer_slice_pitch = buffer_slice_pitch * @sizeOf(T),
                    .host_row_pitch = host_row_pitch * @sizeOf(T),
                    .host_slice_pitch = host_slice_pitch * @sizeOf(T),
                    .src = std.mem.sliceAsBytes(host),
                },
            });
        }

        /// Copies a 3D rectangular region of the buffer into `host`.
        ///
        /// The region is described in elements. `region` is `(width, height,
        /// depth)`: how many elements to copy along each axis. `buffer_origin`
        /// and `host_origin` are `(x, y, z)` offsets giving the lower corner
        /// of the region inside the buffer and `host` respectively.
        ///
        /// Pitches are the stride, in elements, between consecutive rows and
        /// consecutive 2D slices of a 3D layout:
        /// - `buffer_row_pitch`   - stride between row `y` and `y + 1` in the
        ///   buffer.
        /// - `buffer_slice_pitch` - stride between slice `z` and `z + 1` in the
        ///   buffer.
        /// - `host_row_pitch`   - stride between row `y` and `y + 1` in `host`.
        /// - `host_slice_pitch` - stride between slice `z` and `z + 1` in `host`.
        ///
        /// Each pitch must be at least `region[0]` (row) or `region[0] *
        /// region[1]` (slice); larger values leave padding between rows or
        /// slices. Pitches are expressed in elements and converted to bytes
        /// internally. The operation is enqueued on `cq` and returns
        /// immediately without waiting.
        ///
        /// Depending on the backend, the contents of `host` may or may not be
        /// filled immediately, so the caller is responsible for calling
        /// `cq.wait()` and keeping `host` alive for the whole lifetime of the
        /// execution to avoid any issue.
        ///
        /// `cq`                - Command queue used to schedule the read.
        /// `buffer_origin`     - `(x, y, z)` offset of the region's lower corner
        ///                       inside the buffer, in elements.
        /// `host_origin`       - `(x, y, z)` offset of the region's lower corner
        ///                       inside `host`, in elements.
        /// `region`            - `(width, height, depth)` of the region, in
        ///                       elements.
        /// `buffer_row_pitch`  - Row stride inside the buffer, in elements.
        /// `buffer_slice_pitch`- Slice stride inside the buffer, in elements.
        /// `host_row_pitch`   - Row stride inside `host`, in elements.
        /// `host_slice_pitch`  - Slice stride inside `host`, in elements.
        /// `host`              - Destination slice backing the host-side 3D layout.
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
        ) CommandQueue.Error!void {
            try cq.enqueue(.{
                .read_rect = .{
                    .buf = @ptrCast(self),
                    .buffer_origin = buffer_origin,
                    .host_origin = host_origin,
                    .region = region,
                    .buffer_row_pitch = buffer_row_pitch * @sizeOf(T),
                    .buffer_slice_pitch = buffer_slice_pitch * @sizeOf(T),
                    .host_row_pitch = host_row_pitch * @sizeOf(T),
                    .host_slice_pitch = host_slice_pitch * @sizeOf(T),
                    .dst = std.mem.sliceAsBytes(host),
                },
            });
        }

        /// Fills a contiguous range of the buffer with a repeating `pattern`.
        ///
        /// `len` elements starting at element `offset` are overwritten with
        /// repetitions of `pattern`. The pattern is tiled end-to-end across the
        /// target range, so the total byte length must be a whole multiple of
        /// the pattern byte length. The operation is enqueued on `cq` and
        /// returns immediately without waiting.
        ///
        /// Depending on the backend, `pattern` may or may not be copied
        /// immediately, so the caller is responsible for calling `cq.wait()`
        /// and keeping `pattern` alive for the whole lifetime of the execution
        /// to avoid any issue.
        ///
        /// `cq`      - Command queue used to schedule the fill.
        /// `offset`  - Element index (in elements, not bytes) where the fill
        ///              starts.
        /// `len`     - Number of elements to fill.
        /// `pattern` - Pattern slice; its length determines the tiling period.
        pub fn fill(self: *Self, cq: *CommandQueue, offset: usize, len: usize, pattern: []const T) CommandQueue.Error!void {
            try cq.enqueue(.{
                .fill = .{
                    .buf = @ptrCast(self),
                    .offset = offset * @sizeOf(T),
                    .len = len * @sizeOf(T),
                    .pattern = std.mem.sliceAsBytes(pattern),
                },
            });
        }

        /// Fills a 3D rectangular region of the buffer with a repeating `pattern`.
        ///
        /// The region is described in elements. `region` is `(width, height,
        /// depth)`: how many elements to fill along each axis. `buffer_origin`
        /// is the `(x, y, z)` offset of the region's lower corner inside the
        /// buffer. The pattern is tiled end-to-end across the region's elements
        /// in row-major order, so the region's total byte size must be a whole
        /// multiple of the pattern byte length.
        ///
        /// Pitches are the stride, in elements, between consecutive rows and
        /// consecutive 2D slices of the buffer's 3D layout:
        /// - `buffer_row_pitch`   - stride between row `y` and `y + 1`.
        /// - `buffer_slice_pitch` - stride between slice `z` and `z + 1`.
        ///
        /// Each pitch must be at least `region[0]` (row) or `region[0] *
        /// region[1]` (slice); larger values leave padding between rows or
        /// slices. Pitches are expressed in elements and converted to bytes
        /// internally. The operation is enqueued on `cq` and returns
        /// immediately without waiting.
        ///
        /// Depending on the backend, `pattern` may or may not be copied
        /// immediately, so the caller is responsible for calling `cq.wait()`
        /// and keeping `pattern` alive for the whole lifetime of the execution
        /// to avoid any issue.
        ///
        /// `cq`                - Command queue used to schedule the fill.
        /// `buffer_origin`     - `(x, y, z)` offset of the region's lower corner
        ///                       inside the buffer, in elements.
        /// `region`            - `(width, height, depth)` of the region, in
        ///                       elements.
        /// `buffer_row_pitch`  - Row stride inside the buffer, in elements.
        /// `buffer_slice_pitch`- Slice stride inside the buffer, in elements.
        /// `pattern`           - Pattern slice; its length determines the tiling
        ///                       period.
        pub fn fillRect(
            self: *Self,
            cq: *CommandQueue,
            buffer_origin: [3]usize,
            region: [3]usize,
            buffer_row_pitch: usize,
            buffer_slice_pitch: usize,
            pattern: []const T,
        ) CommandQueue.Error!void {
            try cq.enqueue(.{
                .fill_rect = .{
                    .buf = @ptrCast(self),
                    .buffer_origin = buffer_origin,
                    .region = region,
                    .buffer_row_pitch = buffer_row_pitch * @sizeOf(T),
                    .buffer_slice_pitch = buffer_slice_pitch * @sizeOf(T),
                    .pattern = std.mem.sliceAsBytes(pattern),
                },
            });
        }

        /// Copies `len` elements from `src` into `self`.
        ///
        /// The copy is enqueued on `cq`; the queue is not waited on, so the
        /// caller may overlap subsequent work with this transfer.
        ///
        /// `src`        - Source buffer; must outlive the queued copy.
        /// `cq`         - Command queue used to schedule the copy.
        /// `src_offset` - Element index (in elements, not bytes) where the
        ///                source range starts in `src`.
        /// `dst_offset` - Element index (in elements, not bytes) where the
        ///                destination range starts in `self`.
        /// `len`        - Number of elements to copy.
        pub fn copyFrom(
            self: *Self,
            src: *Self,
            cq: *CommandQueue,
            src_offset: usize,
            dst_offset: usize,
            len: usize,
        ) CommandQueue.Error!void {
            try cq.enqueue(.{
                .copy = .{
                    .src_buf = @ptrCast(src),
                    .dst_buf = @ptrCast(self),
                    .src_offset = src_offset * @sizeOf(T),
                    .dst_offset = dst_offset * @sizeOf(T),
                    .len = len * @sizeOf(T),
                },
            });
        }

        /// Copies a 3D rectangular region from `src` into `self`.
        ///
        /// The region is described in elements. `region` is `(width, height,
        /// depth)`: how many elements to copy along each axis. `src_origin` and
        /// `dst_origin` are `(x, y, z)` offsets giving the lower corner of the
        /// region inside `src` and `self` respectively.
        ///
        /// Pitches are the stride, in elements, between consecutive rows and
        /// consecutive 2D slices of a 3D layout:
        /// - `src_row_pitch`   - stride between row `y` and `y + 1` in `src`.
        /// - `src_slice_pitch` - stride between slice `z` and `z + 1` in `src`.
        /// - `dst_row_pitch`   - stride between row `y` and `y + 1` in `self`.
        /// - `dst_slice_pitch` - stride between slice `z` and `z + 1` in `self`.
        ///
        /// Each pitch must be at least `region[0]` (row) or `region[0] *
        /// region[1]` (slice); larger values leave padding between rows or
        /// slices. Pitches are expressed in elements and converted to bytes
        /// internally. The copy is enqueued on `cq`; the queue is not waited on.
        ///
        /// `src`            - Source buffer; must outlive the queued copy.
        /// `cq`             - Command queue used to schedule the copy.
        /// `src_origin`     - `(x, y, z)` offset of the region's lower corner
        ///                     inside `src`, in elements.
        /// `dst_origin`     - `(x, y, z)` offset of the region's lower corner
        ///                     inside `self`, in elements.
        /// `region`         - `(width, height, depth)` of the region, in
        ///                     elements.
        /// `src_row_pitch`  - Row stride inside `src`, in elements.
        /// `src_slice_pitch`- Slice stride inside `src`, in elements.
        /// `dst_row_pitch`  - Row stride inside `self`, in elements.
        /// `dst_slice_pitch`- Slice stride inside `self`, in elements.
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
        ) CommandQueue.Error!void {
            try cq.enqueue(.{
                .copy_rect = .{
                    .src_buf = @ptrCast(src),
                    .dst_buf = @ptrCast(self),
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

        const Self = @This();
    };
}