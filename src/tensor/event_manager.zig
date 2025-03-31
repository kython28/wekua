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

    prev_events: []cl.event.cl_event,
    user_callback: ?UserCallback,

    pub fn init(
        allocator: std.mem.Allocator,
        events_array: []const ?[]const cl.event.cl_event,
        user_callback: ?UserCallback,
    ) !*EventsSet {
        if (events_array.len == 0) return error.EventsArrayEmpty;

        var total_events: usize = 0;
        for (events_array) |events| {
            if (events) |v| {
                total_events += v.len;
            }
        }

        const buf_len = @sizeOf(EventsSet) + @sizeOf(cl.event.cl_event) * total_events;
        const buf = try allocator.alloc(u8, buf_len);
        errdefer allocator.free(buf);

        const set: *EventsSet = @alignCast(@ptrCast(buf.ptr));
        const prev_events = @as(
            [*]cl.event.cl_event,
            @alignCast(@ptrCast(buf.ptr + @sizeOf(EventsSet))),
        )[0..total_events];

        set.* = .{
            .allocator = allocator,
            .buf_len = buf_len,
            .prev_events = prev_events,
            .user_callback = user_callback,
        };

        var offset: usize = 0;
        for (events_array) |v| {
            const events = v orelse continue;
            @memcpy(prev_events[offset..(offset + events.len)], events);
            offset += events.len;
        }

        return set;
    }

    pub inline fn getPrevEvents(self: *const EventsSet) ?[]const cl.event.cl_event {
        const prev_events = self.prev_events;
        if (prev_events.len == 0) return null;

        return prev_events;
    }

    pub fn appendNewEvent(
        self: *EventsSet,
        comptime T: type,
        add_destructor_callback: bool,
        new_ops: []const Operation,
        tensors: []const *Tensor(T),
        prev_cl_events: ?[]const cl.event.cl_event,
        new_cl_event: cl.event.cl_event,
    ) !void {
        if (new_ops.len == 0) {
            return error.InvalidOperations;
        }

        if (new_ops.len != tensors.len) {
            @panic("`new_ops` and `tensors` arrays have different lenghts");
        }

        var ref_counter: usize = @intFromBool(!add_destructor_callback);
        var events_added: usize = 0;
        errdefer {
            for (events_added..ref_counter) |_| {
                cl.event.release(new_cl_event);
            }
        }

        while (ref_counter < new_ops.len) {
            try cl.event.retain(new_cl_event);
            ref_counter += 1;
        }

        var event: *Event = undefined;
        for (new_ops, tensors) |new_op, tensor| {
            event = try tensor.events_manager.appendNewEvent(
                new_op,
                prev_cl_events,
                new_cl_event,
                null,
            );

            events_added += 1;
        }

        if (add_destructor_callback) {
            try event.appendCallback(.{ .func = multipleCompletionEventCallback, .data = self });
            cl.event.release(new_cl_event);
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
    events: [MaxEventsPerSet]cl.event.cl_event,
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
        event: cl.event.cl_event,
        user_callback: ?UserCallback,
    ) !AppendResult {
        const current_operation = self.operation;
        const events_count = self.events_count;

        if (current_operation == .none) {
            self.operation = operation;
        } else if (operation != current_operation or checkFull(current_operation, events_count)) {
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

        if (checkFull(operation, events_count + 1)) return AppendResult.success_and_full;

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

    pub inline fn checkFull(operation: Operation, events_count: usize) bool {
        return switch (operation) {
            .read, .partial_write => (events_count == MaxEventsPerSet),
            .write => (events_count == 1),
            .none => false,
        };
    }

    pub inline fn full(self: *const Event) bool {
        return checkFull(self.operation, self.events_count);
    }

    pub inline fn toSlice(self: *const Event) []const cl.event.cl_event {
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
        try cl.event.wait_for_many(self.toSlice());
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

    prev_events: [MaxEventsPerSet * 2]cl.event.cl_event,
    prev_events_len: MaxEventsPerSetInt,

    events: [EventsBatchLenght]Event,
    events_num: u8,

    pub fn init(self: *EventsBatch, allocator: std.mem.Allocator, prev_events: ?[]const cl.event.cl_event) !void {
        self.allocator = allocator;

        if (prev_events) |pv| {
            if (pv.len > MaxEventsPerSet * 2) return error.EventsArrayTooLong;

            var index: usize = 0;
            errdefer {
                for (pv[0..index]) |e| {
                    cl.event.release(e);
                }
            }

            while (index < pv.len) : (index += 1) {
                try cl.event.retain(pv[index]);
            }

            @memcpy(self.prev_events[0..pv.len], pv);
            self.prev_events_len = @intCast(pv.len);
        } else {
            self.prev_events_len = 0;
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

    pub inline fn getPrevEvents(self: *const EventsBatch) ?[]const cl.event.cl_event {
        const prev_events_len = self.prev_events_len;
        if (prev_events_len > 0) {
            return self.prev_events[0..@intCast(prev_events_len)];
        }
        return null;
    }

    pub inline fn clear(self: *EventsBatch) void {
        for (self.prev_events[0..@intCast(self.prev_events_len)]) |e| {
            cl.event.release(e);
        }

        for (self.events[0..self.events_num]) |*event| {
            event.clear();
        }
    }

    pub fn restart(self: *EventsBatch, new_prev_events: ?[]const cl.event.cl_event) !void {
        for (self.prev_events[0..@intCast(self.prev_events_len)]) |e| {
            cl.event.release(e);
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
        }else if (events_num < EventsBatchLenght) {
            if (self.events[events_num].operation != .none) {
                events_num += 1;
            }
        }

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

allocator: std.mem.Allocator,
batch: *EventsBatch,
events_releaser_queue: *events_releaser.EventsBatchQueue,

pub fn init(self: *TensorEventManager, allocator: std.mem.Allocator, queue: *events_releaser.EventsBatchQueue) !void {
    self.allocator = allocator;

    const batch = try allocator.create(EventsBatch);
    errdefer allocator.destroy(batch);

    try batch.init(allocator, null);

    self.batch = batch;
    self.events_releaser_queue = queue;
}

pub fn deinit(self: *TensorEventManager) void {
    self.batch.waitForPendingEvents();
    self.batch.release();
}

pub fn getPrevEvents(self: *TensorEventManager, new_op: Operation) ?[]const cl.event.cl_event {
    if (new_op == .none) unreachable;

    const batch = self.batch;

    const events_num = batch.events_num;
    if (events_num == EventsBatchLenght) {
        const event: *Event = &batch.events[events_num - 1];
        return event.toSlice();
    }

    const event: *Event = &batch.events[events_num];
    switch (event.operation) {
        .read => switch (new_op) {
            .write, .partial_write => {
                batch.events_num += 1;
                return event.toSlice();
            },
            .read => {},
            else => unreachable,
        },
        .write => unreachable,
        .partial_write => {
            batch.events_num += 1;
            return event.toSlice();
        },
        .none => {},
    }

    if (events_num == 0) {
        return batch.getPrevEvents();
    }

    const prev_event = &batch.events[events_num - 1];
    return prev_event.toSlice();
}

pub fn concat(
    allocator: std.mem.Allocator,
    events_array: []const ?[]const cl.event.cl_event,
) !?[]cl.event.cl_event {
    var total_events: usize = 0;
    for (events_array) |events| {
        if (events) |v| {
            total_events += v.len;
        }
    }

    if (total_events == 0) return null;

    const new_array = try allocator.alloc(cl.event.cl_event, total_events);
    errdefer allocator.free(new_array);

    var offset: usize = 0;
    for (events_array) |v| {
        const events = v orelse continue;
        @memcpy(new_array[offset..(offset + events.len)], events);
        offset += events.len;
    }

    return new_array;
}

fn multipleCompletionEventCallback(user_data: ?*anyopaque) void {
    const set: *EventsSet = @alignCast(@ptrCast(user_data.?));
    const user_callback = set.user_callback;
    if (user_callback) |v| {
        v.func(v.data);
    }

    set.release();
}

fn getNewBatch(self: *TensorEventManager, prev_events: ?[]const cl.event.cl_event) !*EventsBatch {
    const old_batch = self.batch;

    const allocator = self.allocator;
    const new_batch = try allocator.create(EventsBatch);
    errdefer allocator.destroy(new_batch);

    try new_batch.init(allocator, prev_events);
    errdefer new_batch.release();

    self.batch = new_batch;
    errdefer self.batch = old_batch;

    try self.events_releaser_queue.put(old_batch);
    return new_batch;
}

pub fn appendNewEvent(
    self: *TensorEventManager,
    new_op: Operation,
    prev_cl_events: ?[]const cl.event.cl_event,
    new_cl_event: cl.event.cl_event,
    user_callback: ?UserCallback,
) !*Event {
    var batch = self.batch;

    var events_num = batch.events_num;
    if (events_num == EventsBatchLenght) {
        batch = try self.getNewBatch(prev_cl_events);
        events_num = 0;
    }

    var event: *Event = &batch.events[events_num];
    loop: switch (try event.append(new_op, new_cl_event, user_callback)) {
        .success => {},
        .full => {
            events_num += 1;
            if (events_num == EventsBatchLenght) {
                batch.events_num = EventsBatchLenght;

                batch = try self.getNewBatch(prev_cl_events);
                events_num = 0;
            }

            event = &batch.events[events_num];
            const new_result = try event.append(new_op, new_cl_event, user_callback);
            continue :loop new_result;
        },
        .success_and_full => {
            events_num += 1;
        },
    }

    batch.events_num = events_num;
    return event;
}

const TensorEventManager = @This();
