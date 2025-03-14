const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");

pub const UserCallback = struct { func: *const fn (user_data: ?*anyopaque) void, data: ?*anyopaque };

pub const UserCallbackArray = std.ArrayList(UserCallback);

pub const Operation = enum {
    write,
    read,
    none,
};

pub const AppendResult = enum { full, success_and_full, success };

const MaxEventsPerSet = switch (builtin.mode) {
    .Debug => 8,
    .ReleaseFast, .ReleaseSafe => 256,
    .ReleaseSmall => 64,
};

const MaxEventsPerSetInt = switch (builtin.mode) {
    .Debug, .ReleaseSmall => u8,
    .ReleaseFast, .ReleaseSafe => u16,
};

const Event = struct {
    operation: Operation,

    callbacks: UserCallbackArray,
    events: [MaxEventsPerSet]cl.event.cl_event,
    events_count: MaxEventsPerSetInt,
    events_finalized: MaxEventsPerSetInt,

    index: u8,

    pub fn init(self: *Event, index: usize, allocator: std.mem.Allocator) void {
        self.operation = .none;
        self.callbacks = UserCallbackArray.init(allocator);
        self.events_count = 0;
        self.events_finalized = 0;
        self.index = @intCast(index);
    }

    pub inline fn appendNewEvent(
        self: *Event,
        operation: Operation,
        event: cl.event.cl_event,
        user_callback: ?UserCallback,
    ) !AppendResult {
        const current_operation = self.operation;
        if (current_operation == .none) {
            self.operation = operation;
        } else if (operation != current_operation or self.full() or self.finalized()) {
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

        self.events[self.events_count] = event;
        self.events_count += 1;

        if (self.full()) return AppendResult.success_and_full;

        return AppendResult.success;
    }

    pub inline fn pop(self: *Event, had_callback: bool) void {
        self.events_count -= 1;
        if (self.events_count == 0) {
            self.operation = .none;
        }

        if (had_callback) {
            _ = self.callbacks.pop();
        }
    }

    pub inline fn full(self: *const Event) bool {
        return switch (self.operation) {
            .read => (self.events_count == MaxEventsPerSet),
            .write => (self.events_count == 1),
            .none => false,
        };
    }

    pub inline fn finalized(self: *const Event) bool {
        return (self.events_count == self.events_finalized);
    }

    pub inline fn toSlice(self: *const Event) []const cl.event.cl_event {
        return self.events[0..self.events_count];
    }

    pub fn execute_callbacks(self: *const Event) void {
        for (self.callbacks.items) |*callback| {
            callback.func(callback.data);
        }
    }

    pub inline fn waitForEvents(self: *const Event) !void {
        try cl.event.wait_for_many(self.toSlice());
    }

    pub fn clear(self: *Event) void {
        for (self.events[0..self.events_count]) |event| {
            cl.event.release(event);
        }

        self.operation = .none;
        self.events_count = 0;
        self.events_finalized = 0;

        self.callbacks.deinit();
    }
};

const EventsBatchLenght = switch (builtin.mode) {
    .Debug => 4,
    .ReleaseSafe, .ReleaseFast => 128,
    .ReleaseSmall => 16,
};

const EventsBatch = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,

    prev_events: [MaxEventsPerSet * 2]cl.event.cl_event,
    prev_events_len: MaxEventsPerSetInt,

    events: [EventsBatchLenght]Event,
    events_num: u8,
    events_finalized: u8,

    disattached: bool,

    pub fn init(self: *EventsBatch, allocator: std.mem.Allocator, prev_events: ?[]const cl.event.cl_event) !void {
        self.allocator = allocator;
        self.mutex = .{};

        if (prev_events) |pv| {
            if (pv.len > MaxEventsPerSet * 2) return error.EventsArrayTooLong;

            for (pv) |e| try cl.event.retain(e);

            @memcpy(self.prev_events[0..pv.len], pv);
            self.prev_events_len = @intCast(pv.len);
        } else {
            self.prev_events_len = 0;
        }

        for (&self.events, 0..) |*e, index| {
            e.init(index, allocator);
        }

        self.events_num = 0;
        self.events_finalized = 0;

        self.disattached = false;
    }

    pub inline fn empty(self: *const EventsBatch) bool {
        return (self.events_num == 0);
    }

    pub inline fn finalized(self: *const EventsBatch) bool {
        return (self.events_num == self.events_finalized);
    }

    pub inline fn full(self: *const EventsBatch) bool {
        return (self.events_num == EventsBatchLenght);
    }

    pub inline fn markEventAsFinalized(self: *EventsBatch) bool {
        const events_finalized = self.events_finalized + 1;
        self.events_finalized = events_finalized;

        return (self.events_num == events_finalized);
    }

    pub inline fn getPrevEvents(self: *const EventsBatch) ?[]const cl.event.cl_event {
        if (self.prev_events_len > 0) {
            return self.prev_events[0..@intCast(self.prev_events_len)];
        }
        return null;
    }

    pub fn clear(self: *EventsBatch) void {
        for (self.prev_events[0..@intCast(self.prev_events_len)]) |e| {
            cl.event.release(e);
        }

        for (self.events[0..self.events_num]) |*event| {
            event.clear();
        }

        self.prev_events_len = 0;

        self.events_num = 0;
        self.events_finalized = 0;
    }

    pub fn release(self: *EventsBatch) void {
        self.mutex.lock();
        while (!self.finalized()) {
            const events_num = self.events_num;
            self.mutex.unlock();
            defer self.mutex.lock();

            for (self.events[0..events_num]) |*event| {
                event.waitForEvents() catch |err| {
                    std.debug.panic("Unexpected error while waiting for events finalization: {s}", .{@errorName(err)});
                };
            }
        }
        self.mutex.unlock();

        self.clear();
        self.allocator.destroy(self);
    }
};

allocator: std.mem.Allocator,
batch: *EventsBatch,

pub fn init(self: *TensorEventManager, allocator: std.mem.Allocator) !void {
    self.allocator = allocator;

    const batch = try allocator.create(EventsBatch);
    errdefer allocator.destroy(batch);

    try batch.init(allocator, null);

    self.batch = batch;
}

pub fn deinit(self: *TensorEventManager) void {
    self.batch.release();
}

pub fn getPrevEvents(self: *TensorEventManager, new_op: Operation) ?[]const cl.event.cl_event {
    if (new_op == .none) unreachable;

    const batch = self.batch;
    batch.mutex.lock();
    defer batch.mutex.unlock();

    if (batch.empty()) {
        return batch.getPrevEvents();
    }

    const event: *Event = &batch.events[batch.events_num - 1];
    return event.toSlice();
}

fn singleCompletionEventCallback(
    _: cl.event.cl_event,
    event_command_status: i32,
    user_data: ?*anyopaque,
) callconv(.C) void {
    const event_status: cl.event.enums.execution_status = @enumFromInt(event_command_status);
    if (event_status != .complete) unreachable;

    const data: *Event = @alignCast(@ptrCast(user_data.?));
    const ptr: usize = @intFromPtr(data) - @offsetOf(EventsBatch, "events") - @as(usize, data.index) * @sizeOf(Event);
    const batch: *EventsBatch = @ptrFromInt(ptr);

    batch.mutex.lock();

    data.events_finalized += 1;
    if (data.finalized()) {
        data.execute_callbacks();

        if (!data.full()) {
            batch.events_num += 1;
        }

        if (batch.markEventAsFinalized() and batch.disattached) {
            batch.release();
            return;
        }
    }

    batch.mutex.unlock();
}

fn getNewBatch(self: *TensorEventManager, prev_events: ?[]const cl.event.cl_event) !*EventsBatch {
    const old_batch = self.batch;

    const allocator = self.allocator;
    const new_batch = try allocator.create(EventsBatch);
    errdefer allocator.destroy(new_batch);

    try new_batch.init(allocator, prev_events);

    self.batch = new_batch;

    old_batch.disattached = true;
    old_batch.mutex.unlock();

    new_batch.mutex.lock();
    return new_batch;
}

fn addEventToBatch(
    self: *TensorEventManager,
    new_op: Operation,
    prev_cl_events: ?[]const cl.event.cl_event,
    new_cl_event: cl.event.cl_event,
    user_callback: ?UserCallback,
    comptime register_callback: bool,
) !*Event {
    var batch = self.batch;
    batch.mutex.lock();
    defer batch.mutex.unlock();

    if (batch.finalized()) {
        batch.clear();
    } else if (batch.full()) {
        batch = try self.getNewBatch(prev_cl_events);
    }

    var events_num = batch.events_num;
    var event: *Event = &batch.events[events_num];
    loop: switch (try event.appendNewEvent(new_op, new_cl_event, user_callback)) {
        .success => {},
        .full => {
            events_num += 1;
            if (events_num == EventsBatchLenght) {
                batch.events_num = EventsBatchLenght;

                batch = try self.getNewBatch(prev_cl_events);
                events_num = 0;
            }

            event = &batch.events[events_num];
            const new_result = try event.appendNewEvent(new_op, new_cl_event, user_callback);
            continue :loop new_result;
        },
        .success_and_full => {
            events_num += 1;
        },
    }

    batch.events_num = events_num;
    errdefer {
        if (event.operation == .none or !event.full()) {
            batch.events_num -= 1;
        }
    }
    errdefer event.pop(user_callback != null);

    if (register_callback) {
        batch.mutex.unlock();
        defer batch.mutex.lock();

        try cl.event.set_callback(new_cl_event, .complete, &singleCompletionEventCallback, event);
    }

    return event;
}

pub inline fn appendNewEvent(
    self: *TensorEventManager,
    new_op: Operation,
    prev_cl_events: ?[]const cl.event.cl_event,
    new_cl_event: cl.event.cl_event,
    user_callback: ?UserCallback,
) !void {
    _ = try self.addEventToBatch(new_op, prev_cl_events, new_cl_event, user_callback, true);
}

const TensorEventManager = @This();
