const std = @import("std");
const builtin = @import("builtin");

const Workers = @import("workers.zig");
const CommandQueue = @import("command_queue.zig");

const MIN_BUFFER_SIZE_PER_WORKER = 25 * 1024 * 1024;

const CANARY_VALUE: usize = switch (builtin.cpu.arch) {
    .x86_64, .aarch64 => 0xDEADBEEFCAFEBABE,
    else => 0xDEADBEEF,
};

pub const Canary = usize;
pub const BufferHeader = extern struct {
    len: usize,
    canary: Canary,
};

pub fn alloc(allocator: std.mem.Allocator, len: usize) ?[*]u8 {
    const buf = allocator.alloc(u8, len + @sizeOf(BufferHeader) + @sizeOf(Canary)) catch return null;
    const hdr: *BufferHeader = std.mem.bytesAsValue(BufferHeader, buf[0..@sizeOf(BufferHeader)]);
    hdr.* = BufferHeader{
        .len = len,
        .canary = CANARY_VALUE,
    };

    const offset = @sizeOf(BufferHeader) + len;
    const prol_canary: *usize = std.mem.bytesAsValue(usize, buf[offset .. offset + @sizeOf(usize)]);
    prol_canary.* = CANARY_VALUE;

    return buf.ptr;
}

pub fn free(allocator: std.mem.Allocator, buf: [*]u8) void {
    const hdr: *BufferHeader = std.mem.bytesAsValue(BufferHeader, buf[0..@sizeOf(BufferHeader)]);
    if (hdr.canary != CANARY_VALUE) {
        @panic("free: header canary mismatch (buffer header corrupted or wrong pointer)");
    }

    const len = hdr.len;
    const buf_len = @sizeOf(BufferHeader) + len;
    const prol_canary: *usize = std.mem.bytesAsValue(usize, buf[buf_len .. buf_len + @sizeOf(usize)]);
    if (prol_canary.* != CANARY_VALUE) {
        @panic("free: trailer canary mismatch (buffer overrun or wrong length in header)");
    }

    const slice = buf[0 .. buf_len + @sizeOf(usize)];
    allocator.free(slice);
}

fn getSlice(buf: [*]u8) []u8 {
    const hdr: *BufferHeader = std.mem.bytesAsValue(BufferHeader, buf[0..@sizeOf(BufferHeader)]);
    if (hdr.canary != CANARY_VALUE) {
        @panic("getSlice: header canary mismatch (buffer header corrupted or wrong pointer)");
    }

    const len = hdr.len;
    var offset = @sizeOf(BufferHeader);
    const slice = buf[offset .. offset + len];

    offset += len;
    const prol_canary: *usize = std.mem.bytesAsValue(usize, buf[offset .. offset + @sizeOf(usize)]);
    if (prol_canary.* != CANARY_VALUE) {
        @panic("getSlice: trailer canary mismatch (buffer overrun or wrong length in header)");
    }

    return slice;
}

inline fn calculateNumberOfChunks(len: usize) usize {
    const aligned_len = (len - len % MIN_BUFFER_SIZE_PER_WORKER) + MIN_BUFFER_SIZE_PER_WORKER;
    return aligned_len / MIN_BUFFER_SIZE_PER_WORKER;
}

pub fn prepareReadCommand(
    cmd: *CommandQueue,
    allocator: std.mem.Allocator,
    command: CommandQueue.ReadCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const buf = getSlice(@ptrCast(@alignCast(command.buf)));
    const buf_len = buf.len;
    const dst = command.dst;
    var start = command.offset;
    var end = start + dst.len;
    if (start >= buf_len or end > buf_len) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }

    const src = buf[start..end];

    const number_of_chunks = calculateNumberOfChunks(dst.len);

    const slots = try allocator.alloc(Workers.Slot, number_of_chunks);
    errdefer allocator.free(slots);

    const chunk_size = @min(dst.len, MIN_BUFFER_SIZE_PER_WORKER);

    start = 0;
    end = chunk_size;
    for (slots) |*s| {
        const args = s.args[0..4];

        const slot_src = src[start..end];
        const slot_dst = dst[start..end];

        args[0] = @intFromPtr(slot_src.ptr);
        args[1] = slot_src.len;
        args[2] = @intFromPtr(slot_dst.ptr);
        args[3] = slot_dst.len;

        start += chunk_size;
        end = @min(end + chunk_size, dst.len);

        s.command_queue = cmd;
        s.callback = transferKernel;
    }

    return slots;
}

pub fn prepareWriteCommand(
    cmd: *CommandQueue,
    allocator: std.mem.Allocator,
    command: CommandQueue.WriteCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const buf = getSlice(@ptrCast(@alignCast(command.buf)));
    const buf_len = buf.len;
    const src = command.src;
    var start = command.offset;
    var end = start + src.len;
    if (start >= buf_len or end > buf_len) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }

    const dst = buf[start..end];

    const number_of_chunks = calculateNumberOfChunks(src.len);

    const slots = try allocator.alloc(Workers.Slot, number_of_chunks);
    errdefer allocator.free(slots);

    const chunk_size = @min(src.len, MIN_BUFFER_SIZE_PER_WORKER);

    start = 0;
    end = chunk_size;
    for (slots) |*s| {
        const args = s.args[0..4];

        const slot_src = src[start..end];
        const slot_dst = dst[start..end];

        args[0] = @intFromPtr(slot_src.ptr);
        args[1] = slot_src.len;
        args[2] = @intFromPtr(slot_dst.ptr);
        args[3] = slot_dst.len;

        start += chunk_size;
        end = @min(end + chunk_size, src.len);

        s.command_queue = cmd;
        s.callback = transferKernel;
    }

    return slots;
}

pub fn prepareReadRectCommand(
    cmd: *CommandQueue,
    allocator: std.mem.Allocator,
    command: CommandQueue.ReadRectCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const buf = getSlice(@ptrCast(@alignCast(command.buf)));

    const region = command.region;
    const buffer_origin = command.buffer_origin;
    const host_origin = command.host_origin;

    if (!fitsInBuffer(buf.len, buffer_origin, region, command.buffer_row_pitch, command.buffer_slice_pitch)) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }
    if (!fitsInHostBuffer(command.dst.len, host_origin, region, command.host_row_pitch, command.host_slice_pitch)) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }

    const number_of_chunks = calculateNumberOfChunks(region[0]);
    const number_of_slots = number_of_chunks * region[1] * region[2];

    const slots = try allocator.alloc(Workers.Slot, number_of_slots);
    errdefer allocator.free(slots);

    var i: usize = 0;
    var z: usize = 0;
    while (z < region[2]) : (z += 1) {
        var y: usize = 0;
        while (y < region[1]) : (y += 1) {
            var x: usize = 0;
            while (x < region[0]) {
                const row_start = x;
                const row_end_in_region = region[0];
                const row_remaining = row_end_in_region - row_start;
                const chunk_remaining = MIN_BUFFER_SIZE_PER_WORKER;
                const take = @min(row_remaining, chunk_remaining);

                const slot = &slots[i];
                const args = slot.args[0..4];

                const src = buf[buffer_origin[2] * command.buffer_slice_pitch +
                    (buffer_origin[1] + y) * command.buffer_row_pitch +
                    (buffer_origin[0] + row_start) ..][0..take];
                const dst = command.dst[host_origin[2] * command.host_slice_pitch +
                    (host_origin[1] + y) * command.host_row_pitch +
                    (host_origin[0] + row_start) ..][0..take];

                args[0] = @intFromPtr(src.ptr);
                args[1] = src.len;
                args[2] = @intFromPtr(dst.ptr);
                args[3] = dst.len;

                slot.command_queue = cmd;
                slot.callback = transferKernel;

                i += 1;
                x += take;
            }
        }
    }

    return slots;
}

pub fn prepareWriteRectCommand(
    cmd: *CommandQueue,
    allocator: std.mem.Allocator,
    command: CommandQueue.WriteRectCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const buf = getSlice(@ptrCast(@alignCast(command.buf)));

    const region = command.region;
    const buffer_origin = command.buffer_origin;
    const host_origin = command.host_origin;

    if (!fitsInBuffer(buf.len, buffer_origin, region, command.buffer_row_pitch, command.buffer_slice_pitch)) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }
    if (!fitsInHostBuffer(command.src.len, host_origin, region, command.host_row_pitch, command.host_slice_pitch)) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }

    const number_of_chunks = calculateNumberOfChunks(region[0]);
    const number_of_slots = number_of_chunks * region[1] * region[2];

    const slots = try allocator.alloc(Workers.Slot, number_of_slots);
    errdefer allocator.free(slots);

    var i: usize = 0;
    var z: usize = 0;
    while (z < region[2]) : (z += 1) {
        var y: usize = 0;
        while (y < region[1]) : (y += 1) {
            var x: usize = 0;
            while (x < region[0]) {
                const row_start = x;
                const row_end_in_region = region[0];
                const row_remaining = row_end_in_region - row_start;
                const chunk_remaining = MIN_BUFFER_SIZE_PER_WORKER;
                const take = @min(row_remaining, chunk_remaining);

                const slot = &slots[i];
                const args = slot.args[0..4];

                const src = command.src[host_origin[2] * command.host_slice_pitch +
                    (host_origin[1] + y) * command.host_row_pitch +
                    (host_origin[0] + row_start) ..][0..take];
                const dst = buf[buffer_origin[2] * command.buffer_slice_pitch +
                    (buffer_origin[1] + y) * command.buffer_row_pitch +
                    (buffer_origin[0] + row_start) ..][0..take];

                args[0] = @intFromPtr(src.ptr);
                args[1] = src.len;
                args[2] = @intFromPtr(dst.ptr);
                args[3] = dst.len;

                slot.command_queue = cmd;
                slot.callback = transferKernel;

                i += 1;
                x += take;
            }
        }
    }

    return slots;
}

pub fn prepareCopyCommand(
    allocator: std.mem.Allocator,
    cmd: *CommandQueue,
    command: CommandQueue.CopyCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const src_len = command.len;
    const number_of_chunks = calculateNumberOfChunks(src_len);

    const slots = try allocator.alloc(Workers.Slot, number_of_chunks);
    errdefer allocator.free(slots);

    const chunk_size = @min(src_len, MIN_BUFFER_SIZE_PER_WORKER);

    const src_buf = getSlice(@ptrCast(@alignCast(command.src_buf)));
    const dst_buf = getSlice(@ptrCast(@alignCast(command.dst_buf)));

    var start: usize = 0;
    var end: usize = chunk_size;
    for (slots) |*s| {
        const args = s.args[0..4];

        const slot_src = src_buf[command.src_offset + start .. command.src_offset + end];
        const slot_dst = dst_buf[command.dst_offset + start .. command.dst_offset + end];

        args[0] = @intFromPtr(slot_src.ptr);
        args[1] = slot_src.len;
        args[2] = @intFromPtr(slot_dst.ptr);
        args[3] = slot_dst.len;

        start += chunk_size;
        end = @min(end + chunk_size, src_len);

        s.command_queue = cmd;
        s.callback = transferKernel;
    }

    return slots;
}

pub fn prepareCopyRectCommand(
    allocator: std.mem.Allocator,
    cmd: *CommandQueue,
    command: CommandQueue.CopyRectCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const region = command.region;
    const src_origin = command.src_origin;
    const dst_origin = command.dst_origin;

    const src_buf = getSlice(@ptrCast(@alignCast(command.src_buf)));
    const dst_buf = getSlice(@ptrCast(@alignCast(command.dst_buf)));

    if (!fitsInBuffer(src_buf.len, src_origin, region, command.src_row_pitch, command.src_slice_pitch)) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }
    if (!fitsInBuffer(dst_buf.len, dst_origin, region, command.dst_row_pitch, command.dst_slice_pitch)) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }

    const number_of_chunks = calculateNumberOfChunks(region[0]);
    const number_of_slots = number_of_chunks * region[1] * region[2];

    const slots = try allocator.alloc(Workers.Slot, number_of_slots);
    errdefer allocator.free(slots);

    var i: usize = 0;
    var z: usize = 0;
    while (z < region[2]) : (z += 1) {
        var y: usize = 0;
        while (y < region[1]) : (y += 1) {
            var x: usize = 0;
            while (x < region[0]) {
                const row_start = x;
                const row_end_in_region = region[0];
                const row_remaining = row_end_in_region - row_start;
                const chunk_remaining = MIN_BUFFER_SIZE_PER_WORKER;
                const take = @min(row_remaining, chunk_remaining);

                const slot = &slots[i];
                const args = slot.args[0..4];

                const slot_src = src_buf[src_origin[2] * command.src_slice_pitch +
                    (src_origin[1] + y) * command.src_row_pitch +
                    (src_origin[0] + row_start) ..][0..take];
                const slot_dst = dst_buf[dst_origin[2] * command.dst_slice_pitch +
                    (dst_origin[1] + y) * command.dst_row_pitch +
                    (dst_origin[0] + row_start) ..][0..take];

                args[0] = @intFromPtr(slot_src.ptr);
                args[1] = slot_src.len;
                args[2] = @intFromPtr(slot_dst.ptr);
                args[3] = slot_dst.len;

                slot.command_queue = cmd;
                slot.callback = transferKernel;

                i += 1;
                x += take;
            }
        }
    }

    return slots;
}

pub fn prepareFillCommand(
    cmd: *CommandQueue,
    allocator: std.mem.Allocator,
    command: CommandQueue.FillCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const buf = getSlice(@ptrCast(@alignCast(command.buf)));
    const buf_len = buf.len;
    const pattern = command.pattern;
    const start = command.offset;
    const end = start + command.len;
    if (start >= buf_len or end > buf_len) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }
    if (pattern.len == 0 or command.len % pattern.len != 0) {
        return CommandQueue.EnqueueError.InvalidPitch;
    }

    const number_of_chunks = calculateNumberOfChunks(command.len);

    const slots = try allocator.alloc(Workers.Slot, number_of_chunks);
    errdefer allocator.free(slots);

    const chunk_size = @min(command.len, MIN_BUFFER_SIZE_PER_WORKER);

    const dst_base: [*]u8 = buf.ptr;
    var chunk_start: usize = 0;
    var chunk_end: usize = chunk_size;
    for (slots) |*s| {
        const args = s.args[0..3];

        const dst = dst_base[start + chunk_start .. start + chunk_end];

        args[0] = @intFromPtr(dst.ptr);
        args[1] = dst.len;
        args[2] = @intFromPtr(pattern.ptr);

        chunk_start += chunk_size;
        chunk_end = @min(chunk_end + chunk_size, command.len);

        s.command_queue = cmd;
        s.callback = genericFillKernel;
    }

    return slots;
}

pub fn prepareFillRectCommand(
    cmd: *CommandQueue,
    allocator: std.mem.Allocator,
    command: CommandQueue.FillRectCommand,
) CommandQueue.EnqueueError![]const Workers.Slot {
    const buf = getSlice(@ptrCast(@alignCast(command.buf)));

    const pattern = command.pattern;
    const buffer_origin = command.buffer_origin;
    const region = command.region;

    if (!fitsInBuffer(buf.len, buffer_origin, region, command.buffer_row_pitch, command.buffer_slice_pitch)) {
        return CommandQueue.EnqueueError.OutOfBounds;
    }

    const total_bytes = region[0] * region[1] * region[2];
    if (pattern.len == 0 or total_bytes % pattern.len != 0) {
        return CommandQueue.EnqueueError.InvalidPitch;
    }

    const number_of_chunks = calculateNumberOfChunks(region[0]);
    const number_of_slots = number_of_chunks * region[1] * region[2];

    const slots = try allocator.alloc(Workers.Slot, number_of_slots);
    errdefer allocator.free(slots);

    var i: usize = 0;
    var z: usize = 0;
    while (z < region[2]) : (z += 1) {
        var y: usize = 0;
        while (y < region[1]) : (y += 1) {
            var x: usize = 0;
            while (x < region[0]) {
                const row_start = x;
                const row_end_in_region = region[0];
                const row_remaining = row_end_in_region - row_start;
                const chunk_remaining = MIN_BUFFER_SIZE_PER_WORKER;
                const take = @min(row_remaining, chunk_remaining);

                const slot = &slots[i];
                const args = slot.args[0..3];

                const dst = buf[buffer_origin[2] * command.buffer_slice_pitch +
                    (buffer_origin[1] + y) * command.buffer_row_pitch +
                    (buffer_origin[0] + row_start) ..][0..take];

                args[0] = @intFromPtr(dst.ptr);
                args[1] = dst.len;
                args[2] = @intFromPtr(pattern.ptr);
                args[3] = pattern.len;

                slot.command_queue = cmd;
                slot.callback = genericFillKernel;

                i += 1;
                x += take;
            }
        }
    }

    return slots;
}

fn transferKernel(args: []const usize) void {
    const src: [*]const u8 = @ptrFromInt(args[0]);
    const src_len = args[1];
    const dst: [*]u8 = @ptrFromInt(args[2]);
    const dst_len = args[3];

    @memcpy(dst[0..dst_len], src[0..src_len]);
}

fn genericFillKernel(args: []const usize) void {
    const dst: [*]u8 = @ptrFromInt(args[0]);
    const dst_len = args[1];
    const pattern: [*]const u8 = @ptrFromInt(args[2]);
    const pattern_len = args[3];

    var written: usize = 0;
    while (written < dst_len) {
        const remaining = dst_len - written;
        const copy_len = @min(pattern_len, remaining);
        @memcpy(dst[written .. written + copy_len], pattern[0..copy_len]);
        written += copy_len;
    }
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
