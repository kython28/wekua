const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");

const Batch = @import("batch.zig");

pub const AppendResult = enum { full, success_and_full, success };

pub const UserCallback = struct {
    func: *const fn (user_data: ?*anyopaque) void,
    data: ?*anyopaque,
};
pub const UserCallbackArray = std.ArrayList(UserCallback);

pub const Operation = enum {
    write,
    partial_write,
    read,
    none,
};

const MaxEventsPerSet = switch (builtin.mode) {
    .Debug => 8,
    .ReleaseFast, .ReleaseSafe => 256,
    .ReleaseSmall => 64,
};

const MaxEventsPerSetInt = switch (builtin.mode) {
    .Debug, .ReleaseSmall => u8,
    .ReleaseFast, .ReleaseSafe => u16,
};

operation: Operation,

callbacks: UserCallbackArray,
events: [MaxEventsPerSet]cl.event.Event,
events_count: MaxEventsPerSetInt,

index: u8,

pub fn init(self: *Event, index: usize, allocator: std.mem.Allocator) void {
    self.operation = .none;
    self.callbacks = UserCallbackArray.init(allocator);
    self.events_count = 0;
    self.index = @intCast(index);
}

pub inline fn getParent(self: *Event) *Batch {
    const ptr: usize = @intFromPtr(self) - @offsetOf(Batch, "events") - @as(usize, self.index) * @sizeOf(Event);
    return @ptrFromInt(ptr);
}

pub fn append(
    self: *Event,
    operation: Operation,
    event: cl.event.Event,
    user_callback: ?UserCallback,
) !AppendResult {
    const current_operation = self.operation;
    const events_count = self.events_count;

    if (current_operation == .none) {
        self.operation = operation;
    } else if (operation != current_operation or isFull(current_operation, events_count)) {
        return AppendResult.full;
    }
    errdefer {
        if (current_operation == .none) {
            self.operation = .none;
        }
    }

    if (user_callback) |callback| {
        try self.callbacks.append(callback);
    }

    self.events[events_count] = event;
    self.events_count = events_count + 1;

    if (isFull(operation, events_count + 1)) return AppendResult.success_and_full;

    return AppendResult.success;
}

pub inline fn pop(self: *Event, had_callback: bool) void {
    const events_count = self.events_count;
    self.events_count = events_count - 1;
    if (events_count == 0) {
        self.operation = .none;
    }

    if (had_callback) {
        _ = self.callbacks.pop();
    }
}

pub inline fn isFull(operation: Operation, events_count: usize) bool {
    return switch (operation) {
        .read, .partial_write => (events_count == MaxEventsPerSet),
        .write => (events_count == 1),
        .none => false,
    };
}

pub inline fn full(self: *const Event) bool {
    return isFull(self.operation, self.events_count);
}

pub inline fn toSlice(self: *const Event) []const cl.event.Event {
    return self.events[0..self.events_count];
}

pub inline fn appendCallback(self: *Event, callback: UserCallback) !void {
    try self.callbacks.append(callback);
}

pub inline fn executeCallbacks(self: *const Event) void {
    for (self.callbacks.items) |*callback| {
        callback.func(callback.data);
    }
}

pub inline fn waitForEvents(self: *const Event) !void {
    try cl.event.waitForMany(self.toSlice());
    self.executeCallbacks();
}

pub inline fn clear(self: *Event) void {
    for (self.events[0..self.events_count]) |event| {
        cl.event.release(event);
    }

    self.callbacks.deinit();
}

pub inline fn restart(self: *Event) void {
    self.clear();

    self.operation = .none;
    self.events_count = 0;
}

const Event = @This();
