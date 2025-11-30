const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

const events_releaser = @import("../core/events_releaser.zig");

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

pub const AppendResult = enum { full, success_and_full, success };

const MaxEventsPerSet = switch (builtin.mode) {
    .Debug => 32,
    .ReleaseFast, .ReleaseSafe => 256,
    .ReleaseSmall => 64,
};

const MaxEventsPerSetInt = switch (builtin.mode) {
    .Debug, .ReleaseSmall => u8,
    .ReleaseFast, .ReleaseSafe => u16,
};

pub const EventsSet = struct {
    allocator: std.mem.Allocator,
    buf_len: usize,

    prev_events: ?[]cl.event.Event,
    user_callback: ?UserCallback,

    pub fn init(
        allocator: std.mem.Allocator,
        events_array: []const ?[]const cl.event.Event,
        user_callback: ?UserCallback,
    ) !*EventsSet {
        if (events_array.len == 0) return error.EventsArrayEmpty;

        var total_events: usize = 0;
        for (events_array) |events| {
            if (events) |v| {
                total_events += v.len;
            }
        }

        const buf_len = @sizeOf(EventsSet) + @sizeOf(cl.event.Event) * total_events;
        const buf = try allocator.alloc(u8, buf_len);
        errdefer allocator.free(buf);

        const set: *EventsSet = @alignCast(@ptrCast(buf.ptr));
        const prev_events: ?[]cl.event.Event = blk: {
            if (total_events == 0) break :blk null;

            const array = @as(
                [*]cl.event.Event,
                @alignCast(@ptrCast(buf.ptr + @sizeOf(EventsSet))),
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

    pub inline fn getPrevEvents(self: *const EventsSet) ?[]const cl.event.Event {
        return self.prev_events;
    }

    pub fn appendNewEvent(
        self: *EventsSet,
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
            event = try tensor.event_manager.appendNewEvent(
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

    pub inline fn release(self: *EventsSet) void {
        const buf: []u8 = @as([*]u8, @ptrCast(self))[0..self.buf_len];
        self.allocator.free(buf);
    }
};

const Event = struct {
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

    pub inline fn getParent(self: *Event) *EventsBatch {
        const ptr: usize = @intFromPtr(self) - @offsetOf(EventsBatch, "events") - @as(usize, self.index) * @sizeOf(Event);
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
};

const EventsBatchLenght = switch (builtin.mode) {
    .Debug => 4,
    .ReleaseSafe, .ReleaseFast => 128,
    .ReleaseSmall => 16,
};

pub const EventsBatch = struct {
    allocator: std.mem.Allocator,
    prev_events: ?[]cl.event.Event,

    events: [EventsBatchLenght]Event,
    events_num: u8,

    pub fn init(
        self: *EventsBatch,
        allocator: std.mem.Allocator,
        prev_events: ?[]const cl.event.Event,
    ) !void {
        self.allocator = allocator;

        if (prev_events) |pv| {
            const _prev_events = try allocator.dupe(cl.event.Event, pv);
            errdefer allocator.free(_prev_events);

            var index: usize = 0;
            errdefer {
                for (pv[0..index]) |e| {
                    cl.event.release(e);
                }
            }

            for (pv) |e| {
                try cl.event.retain(e);
                index += 1;
            }

            self.prev_events = _prev_events;
        } else {
            self.prev_events = null;
        }

        errdefer {
            if (self.prev_events) |pv| {
                for (pv) |e| {
                    cl.event.release(e);
                }
                allocator.free(pv);
            }
            self.prev_events = null;
        }

        for (&self.events, 0..) |*e, index| {
            e.init(index, allocator);
        }

        self.events_num = 0;
    }

    pub inline fn empty(self: *const EventsBatch) bool {
        return (self.events_num == 0);
    }

    pub inline fn full(self: *const EventsBatch) bool {
        return (self.events_num == EventsBatchLenght);
    }

    pub inline fn getPrevEvents(self: *const EventsBatch) ?[]const cl.event.Event {
        return self.prev_events;
    }

    pub inline fn clear(self: *EventsBatch) void {
        if (self.prev_events) |prev_events| {
            for (prev_events) |e| {
                cl.event.release(e);
            }
            self.allocator.free(prev_events);
            self.prev_events = null;
        }

        for (self.events[0..self.events_num]) |*event| {
            event.clear();
        }
    }

    pub fn restart(self: *EventsBatch, new_prev_events: ?[]const cl.event.Event) !void {
        if (self.prev_events) |prev_events| {
            for (prev_events) |e| {
                cl.event.release(e);
            }
            self.allocator.free(prev_events);
            self.prev_events = null;
        }

        if (new_prev_events) |pv| {
            if (pv.len > MaxEventsPerSet * 2) return error.EventsArrayTooLong;

            var index: usize = 0;
            errdefer {
                for (pv[0..index]) |e| {
                    cl.event.release(e);
                }
            }

            const prev_events = &self.prev_events;
            while (index < pv.len) : (index += 1) {
                const e = pv[index];

                try cl.event.retain(e);
                prev_events[index] = e;
            }

            self.prev_events_len = @intCast(pv.len);
        }

        for (self.events[0..self.events_num]) |*event| {
            event.restart();
        }

        self.prev_events_len = 0;
        self.events_num = 0;
    }

    pub fn waitForPendingEvents(self: *EventsBatch) void {
        var events_num = self.events_num;
        if (events_num == 0) {
            if (self.events[0].operation == .none) {
                return;
            }

            events_num += 1;
        } else if (events_num < EventsBatchLenght) {
            if (self.events[events_num].operation != .none) {
                events_num += 1;
            }
        }
        self.events_num = events_num;

        for (self.events[0..events_num]) |*event| {
            event.waitForEvents() catch |err| {
                std.debug.panic("Unexpected error while waiting for events finalization: {s}", .{@errorName(err)});
            };
        }
    }

    pub fn release(self: *EventsBatch) void {
        self.clear();
        self.allocator.destroy(self);
    }
};

