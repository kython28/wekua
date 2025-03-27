const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

pub const UserCallback = struct {
    func: *const fn (user_data: ?*anyopaque) void,
    data: ?*anyopaque,
};

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

pub const EventsSet = struct {
    allocator: std.mem.Allocator,
    buf_len: usize,

    events: []*Event,
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

        const buf_len = @sizeOf(EventsSet) + @sizeOf(cl.event.cl_event) * total_events + @sizeOf(*Event) * events_array.len;
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
            .events = @as([*]*Event, @ptrCast(prev_events.ptr + total_events))[0..events_array.len],
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

        var ref_counter: usize = 1;
        errdefer {
            for (1..ref_counter) |_| {
                cl.event.release(new_cl_event);
            }
        }

        while (ref_counter < new_ops.len) {
            try cl.event.retain(new_cl_event);
            ref_counter += 1;
        }

        const events = self.events;

        var events_added: usize = 0;
        errdefer {
            for (events[0..events_added]) |event| {
                const batch = event.getParent();
                batch.mutex.lock();
                defer batch.mutex.unlock();

                const was_full = event.full();
                event.pop(false);
                if (event.operation == .none or was_full) {
                    batch.events_num -= 1;
                }
            }
        }

        for (new_ops, tensors, events) |new_op, tensor, *event| {
            event.* = try addEventToBatch(
                &tensor.events_manager,
                new_op,
                prev_cl_events,
                new_cl_event,
                null,
                false,
            );

            events_added += 1;
        }

        try cl.event.set_callback(new_cl_event, .complete, &multipleCompletionEventCallback, self);
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
    events_finalized: MaxEventsPerSetInt,

    index: u8,

    pub fn init(self: *Event, index: usize, allocator: std.mem.Allocator) void {
        self.operation = .none;
        self.callbacks = UserCallbackArray.init(allocator);
        self.events_count = 0;
        self.events_finalized = 0;
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
        } else if (operation != current_operation or checkFull(current_operation, events_count) or events_count == self.events_finalized) {
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
            .read => (events_count == MaxEventsPerSet),
            .write => (events_count == 1),
            .none => false,
        };
    }

    pub inline fn full(self: *const Event) bool {
        return checkFull(self.operation, self.events_count);
    }

    pub inline fn finalized(self: *const Event) bool {
        return (self.events_count == self.events_finalized);
    }

    pub inline fn toSlice(self: *const Event) []const cl.event.cl_event {
        return self.events[0..self.events_count];
    }

    pub inline fn executeCallbacks(self: *const Event) void {
        for (self.callbacks.items) |*callback| {
            callback.func(callback.data);
        }
    }

    pub inline fn waitForEvents(self: *const Event) !void {
        try cl.event.wait_for_many(self.toSlice());
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
        self.events_finalized = 0;
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
        self.events_finalized = 0;
    }

    pub fn waitForPendingEvents(self: *EventsBatch) void {
        if (self.events_num == 0) {
            if (self.events[0].operation == .none) {
                return;
            }

            self.events_num += 1;
        }

        while (!self.finalized()) {
            const events_num = self.events_num;
            for (self.events[0..events_num]) |*event| {
                if (event.finalized()) continue;

                self.mutex.unlock();
                defer self.mutex.lock();

                event.waitForEvents() catch |err| {
                    std.debug.panic("Unexpected error while waiting for events finalization: {s}", .{@errorName(err)});
                };
            }
        }
    }

    pub fn release(self: *EventsBatch) void {
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
    self.batch.mutex.lock();
    self.batch.waitForPendingEvents();
    self.batch.release();
}

pub fn getPrevEvents(self: *TensorEventManager, new_op: Operation) ?[]const cl.event.cl_event {
    if (new_op == .none) unreachable;

    const batch = self.batch;
    batch.mutex.lock();
    defer batch.mutex.unlock();

    const events_num = batch.events_num;
    if (events_num == EventsBatchLenght) {
        const event: *Event = &batch.events[events_num - 1];
        return event.toSlice();
    }

    const event: *Event = &batch.events[events_num];
    switch (event.operation) {
        .read => switch (new_op) {
            .write => {
                batch.events_num += 1;
                return event.toSlice();
            },
            .read => {},
            else => unreachable,
        },
        .write => unreachable,
        .none => {},
    }

    // if (event.finalized() and events_num == batch.events_finalized) {
    //     batch.events_num += 1;
    //     batch.restart(null) catch |err| {
    //         std.debug.panic("Unexpected error while restarting batch: {s}", .{@errorName(err)});
    //     };
    //     return null;
    // }

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

fn singleCompletionEventCallback(
    _: cl.event.cl_event,
    event_command_status: i32,
    user_data: ?*anyopaque,
) callconv(.C) void {
    const event_status: cl.event.enums.execution_status = @enumFromInt(event_command_status);
    if (event_status != .complete) unreachable;

    const event: *Event = @alignCast(@ptrCast(user_data.?));
    const batch: *EventsBatch = event.getParent();

    batch.mutex.lock();

    const events_finalized = event.events_finalized + 1;
    const events_count = event.events_count;

    event.events_finalized = events_finalized;
    if (events_finalized == events_count) {
        event.executeCallbacks();

        var batch_events_num = batch.events_num;
        if (batch_events_num < EventsBatchLenght and !Event.checkFull(event.operation, events_count) and event == &batch.events[batch_events_num]) {
            batch_events_num += 1;
            batch.events_num = batch_events_num;
        }

        const batch_events_finalized = batch.events_finalized + 1;
        batch.events_finalized = batch_events_finalized;

        if (batch_events_finalized == batch_events_num and batch.disattached) {
            batch.release();
            return;
        }
    }

    batch.mutex.unlock();
}

fn multipleCompletionEventCallback(
    cl_event: cl.event.cl_event,
    event_command_status: i32,
    user_data: ?*anyopaque,
) callconv(.C) void {
    const event_status: cl.event.enums.execution_status = @enumFromInt(event_command_status);
    if (event_status != .complete) unreachable;

    const set: *EventsSet = @alignCast(@ptrCast(user_data.?));
    const events = set.events;
    for (events) |event| {
        @call(
            .always_inline,
            singleCompletionEventCallback,
            .{
                cl_event,
                @intFromEnum(cl.event.enums.execution_status.complete),
                event,
            },
        );
    }
    const user_callback = set.user_callback;
    if (user_callback) |v| {
        v.func(v.data);
    }

    set.release();
}

fn getNewBatch(self: *TensorEventManager, prev_events: ?[]const cl.event.cl_event) !*EventsBatch {
    const old_batch = self.batch;

    if (old_batch.finalized()) {
        try old_batch.restart(prev_events);
        return old_batch;
    }

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
    errdefer {
        event.pop(user_callback != null);
        if (event.operation == .none or !event.full()) {
            batch.events_num -= 1;
        }
    }

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
