const std = @import("std");

const cl = @import("opencl");

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;

const Event = @import("event.zig");

const UserCallback = Event.UserCallback;
const Operation = Event.Operation;

allocator: std.mem.Allocator,
buf_len: usize,

prev_events: ?[]cl.event.Event,
user_callback: ?UserCallback,

pub fn init(
    allocator: std.mem.Allocator,
    events_array: []const ?[]const cl.event.Event,
    user_callback: ?UserCallback,
) !*Set {
    if (events_array.len == 0) return error.EventsArrayEmpty;

    var total_events: usize = 0;
    for (events_array) |events| {
        if (events) |v| {
            total_events += v.len;
        }
    }

    const buf_len = @sizeOf(Set) + @sizeOf(cl.event.Event) * total_events;
    const buf = try allocator.alloc(u8, buf_len);
    errdefer allocator.free(buf);

    const set: *Set = @alignCast(@ptrCast(buf.ptr));
    const prev_events: ?[]cl.event.Event = blk: {
        if (total_events == 0) break :blk null;

        const array = @as(
            [*]cl.event.Event,
            @alignCast(@ptrCast(buf.ptr + @sizeOf(Set))),
        )[0..total_events];

        var offset: usize = 0;
        for (events_array) |v| {
            const events = v orelse continue;
            @memcpy(array[offset..(offset + events.len)], events);
            offset += events.len;
        }

        break :blk array;
    };

    set.* = .{
        .allocator = allocator,
        .buf_len = buf_len,
        .prev_events = prev_events,
        .user_callback = user_callback,
    };

    return set;
}

pub inline fn getPrevEvents(self: *const Set) ?[]const cl.event.Event {
    return self.prev_events;
}

pub fn appendNewEvent(
    self: *Set,
    comptime T: type,
    add_destructor_callback: bool,
    new_ops: []const Operation,
    tensors: []const *Tensor(T),
    new_Event: cl.event.Event,
) !void {
    if (new_ops.len == 0) {
        return error.InvalidOperations;
    }

    if (new_ops.len != tensors.len) {
        @panic("`new_ops` and `tensors` arrays have different lenghts");
    }

    var ref_counter: usize = @intFromBool(!add_destructor_callback);
    var events_added: usize = ref_counter;
    errdefer {
        for (events_added..ref_counter) |_| {
            cl.event.release(new_Event);
        }
    }

    while (ref_counter < new_ops.len) {
        try cl.event.retain(new_Event);
        ref_counter += 1;
    }

    const prev_events = self.prev_events;

    var event: *Event = undefined;
    for (new_ops, tensors) |new_op, tensor| {
        event = try tensor.events.appendNewEvent(
            new_op,
            prev_events,
            new_Event,
            null,
        );

        events_added += 1;
    }

    if (add_destructor_callback) {
        try event.appendCallback(.{ .func = multipleCompletionEventCallback, .data = self });
        cl.event.release(new_Event);
    }
}

pub inline fn release(self: *Set) void {
    const buf: []u8 = @as([*]u8, @ptrCast(self))[0..self.buf_len];
    self.allocator.free(buf);
}

fn multipleCompletionEventCallback(user_data: ?*anyopaque) void {
    const set: *Set = @alignCast(@ptrCast(user_data.?));
    const user_callback = set.user_callback;
    if (user_callback) |v| {
        v.func(v.data);
    }

    set.release();
}

const Set = @This();
