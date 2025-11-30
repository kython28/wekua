const std = @import("std");

const cl = @import("opencl");

const Set = @import("set.zig");
const Batch = @import("batch.zig");
pub const Combined = @import("combined.zig");

const Operation = Set.Operation;
const BatchLength = Batch.Length;

pub const UserCallback = Set.UserCallback;
pub const Errors = Batch.Errors;

allocator: std.mem.Allocator,
batch: *Batch,
batches: Batch.BatchQueue,
can_restart: bool,

pub fn init(
    self: *Events,
    allocator: std.mem.Allocator,
) Errors!void {
    self.allocator = allocator;

    const batch = try Batch.init(allocator, null);
    errdefer batch.deinit();

    self.batch = batch;
    self.batches = Batch.BatchQueue{};
    self.can_restart = false;
}

fn deinitBatches(self: *Events) void {
    const allocator = self.allocator;

    var maybe_node = self.batches.head;
    while (maybe_node) |node| {
        const batch = &node.data;
        maybe_node = node.next;

        batch.deinit(allocator);
    }
}

fn waitForPendingEvents(self: *Events) void {
    var maybe_node = self.batches.head;
    while (maybe_node) |node| {
        const batch = &node.data;
        maybe_node = node.next;

        batch.waitForPendingEvents();
    }
    self.batch.waitForPendingEvents();
}

pub fn deinit(self: *Events) void {
    self.waitForPendingEvents();
    self.deinitBatches();
    self.batch.deinit(self.allocator);
}

fn restart(self: *Events) void {
    self.waitForPendingEvents();
    self.deinitBatches();
    self.batch.clear(self.allocator);
    self.can_restart = false;
}

pub fn getPrevEvents(self: *Events, new_op: Operation) ?[]const cl.event.Event {
    if (new_op == .none) @panic("Invalid operation");

    if (self.can_restart) {
        self.restart();
    }

    const batch = self.batch;

    const events_num = batch.number_of_sets;
    if (events_num == BatchLength) {
        const event: *Set = &batch.sets[events_num - 1];
        return event.toSlice();
    }

    const events_set: *Set = &batch.sets[events_num];
    switch (events_set.operation) {
        .read => switch (new_op) {
            .write, .partial_write => {
                batch.number_of_sets += 1;
                return events_set.toSlice();
            },
            .read => {},
            else => @panic("Invalid operation"),
        },
        .write => @panic("Unexpected writing events set gotten"),
        .partial_write => {
            batch.number_of_sets += 1;
            return events_set.toSlice();
        },
        .none => {},
    }

    if (events_num == 0) {
        return batch.getPrevEvents();
    }

    const prev_event = &batch.sets[events_num - 1];
    return prev_event.toSlice();
}

pub fn concat(
    allocator: std.mem.Allocator,
    events_array: []const ?[]const cl.event.Event,
) error{OutOfMemory}!?[]cl.event.Event {
    var total_events: usize = 0;
    for (events_array) |events| {
        if (events) |v| {
            total_events += v.len;
        }
    }

    if (total_events == 0) return null;

    const new_array = try allocator.alloc(cl.event.Event, total_events);
    errdefer allocator.free(new_array);

    var offset: usize = 0;
    for (events_array) |v| {
        const events = v orelse continue;
        @memcpy(new_array[offset..(offset + events.len)], events);
        offset += events.len;
    }

    return new_array;
}


fn getNewBatch(self: *Events, prev_events: ?[]const cl.event.Event) Errors!*Batch {
    const old_batch = self.batch;

    const allocator = self.allocator;
    const new_batch = try Batch.init(allocator, prev_events);

    self.batch = new_batch;
    old_batch.push(&self.batches);

    return new_batch;
}

pub fn appendNewEvent(
    self: *Events,
    new_op: Operation,
    prev_events: ?[]const cl.event.Event,
    new_event: cl.event.Event,
    user_callback: ?UserCallback,
) Errors!*Set {
    if (self.can_restart) {
        self.restart();
    }

    var batch = self.batch;

    var events_num = batch.number_of_sets;
    if (events_num == BatchLength) {
        batch = try self.getNewBatch(prev_events);
        events_num = 0;
    }

    const allocator = self.allocator;
    var events_set: *Set = &batch.sets[events_num];
    loop: switch (try events_set.append(allocator, new_op, new_event, user_callback)) {
        .success => {},
        .full => {
            events_num += 1;
            if (events_num == BatchLength) {
                batch.number_of_sets = BatchLength;

                batch = try self.getNewBatch(prev_events);
                events_num = 0;
            }

            events_set = &batch.sets[events_num];
            const new_result = try events_set.append(allocator, new_op, new_event, user_callback);
            continue :loop new_result;
        },
        .success_and_full => {
            events_num += 1;
        },
    }

    batch.number_of_sets = events_num;
    return events_set;
}

const Events = @This();

// Tests
const testing = std.testing;
const core = @import("core");

test "Events.init initializes correctly" {
    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    try testing.expect(events.batch.number_of_sets == 0);
    try testing.expect(@intFromPtr(events.events_releaser_queue) == @intFromPtr(&queue));
}

test "Events.getPrevEvents with empty batch and no prev events" {
    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    const result = events.getPrevEvents(.read);
    try testing.expectEqual(null, result);
}

test "Events.getPrevEvents with read operation and existing read operation" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // Add a read event first
    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);

    try cl.event.setUserEventStatus(cl_event1, .complete);

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(testing.allocator, .read, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result1 == .success);

    // Now getPrevEvents with read should return null (no increment)
    const prev_events = events.getPrevEvents(.read);
    try testing.expect(prev_events == null);
    try testing.expect(events.batch.number_of_sets == 0);
}

test "Events.getPrevEvents with read operation and existing write operation transitions" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // Add a read event first
    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);

    try cl.event.setUserEventStatus(cl_event1, .complete);

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(testing.allocator, .read, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result1 == .success);

    // Now getPrevEvents with write should increment and return events
    const prev_events = events.getPrevEvents(.write);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 1);
    try testing.expect(prev_events.?[0] == cl_event1);
    try testing.expect(events.batch.number_of_sets == 1);
}

test "Events.getPrevEvents with partial_write operation transitions correctly" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // Add a partial_write event first
    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);

    try cl.event.setUserEventStatus(cl_event1, .complete);

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(testing.allocator, .partial_write, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result1 == .success);

    // Any new operation should increment and return events
    const prev_events = events.getPrevEvents(.read);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 1);
    try testing.expect(prev_events.?[0] == cl_event1);
    try testing.expect(events.batch.number_of_sets == 1);
}

test "Events.getPrevEvents with full batch returns last event" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // Fill the batch to capacity
    events.batch.number_of_sets = BatchLength;

    // Add an event to the last slot
    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    try cl.event.setUserEventStatus(cl_event, .complete);

    try cl.event.retain(cl_event);
    const result = events.batch.sets[BatchLength - 1].append(testing.allocator, .read, cl_event, null) catch |err| {
        cl.event.release(cl_event);
        return err;
    };
    try testing.expect(result == .success);

    const prev_events = events.getPrevEvents(.write);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 1);
    try testing.expect(prev_events.?[0] == cl_event);
}

test "Events.concat with empty events array returns null" {
    const events_array: []const ?[]const cl.event.Event = &.{};
    const result = try Events.concat(testing.allocator, events_array);
    try testing.expect(result == null);
}

test "Events.concat with all null events returns null" {
    const events_array: []const ?[]const cl.event.Event = &.{ null, null, null };
    const result = try Events.concat(testing.allocator, events_array);
    try testing.expect(result == null);
}

test "Events.concat with mixed null and valid events" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);
    const cl_event3 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event3);

    const events1 = [_]cl.event.Event{ cl_event1, cl_event2 };
    const events2 = [_]cl.event.Event{cl_event3};

    const events_array: []const ?[]const cl.event.Event = &.{ &events1, null, &events2 };

    const result = try Events.concat(testing.allocator, events_array);
    defer if (result) |r| testing.allocator.free(r);

    try testing.expect(result != null);
    try testing.expect(result.?.len == 3);
    try testing.expect(result.?[0] == cl_event1);
    try testing.expect(result.?[1] == cl_event2);
    try testing.expect(result.?[2] == cl_event3);
}

test "Events.concat with single event array" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const events = [_]cl.event.Event{cl_event};
    const events_array: []const ?[]const cl.event.Event = &.{&events};

    const result = try Events.concat(testing.allocator, events_array);
    defer if (result) |r| testing.allocator.free(r);

    try testing.expect(result != null);
    try testing.expect(result.?.len == 1);
    try testing.expect(result.?[0] == cl_event);
}

test "Events.getNewBatch creates new batch and queues old one" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    const old_batch = events.batch;

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const prev_events = [_]cl.event.Event{cl_event};

    const new_batch = try events.getNewBatch(&prev_events);

    try testing.expect(@intFromPtr(new_batch) != @intFromPtr(old_batch));
    try testing.expect(@intFromPtr(events.batch) == @intFromPtr(new_batch));
    try testing.expect(new_batch.prev_events != null);
    try testing.expect(new_batch.prev_events.?.len == 1);

    // Check that old batch was queued
    try testing.expect(!queue.isEmpty());
    const queued_batch = queue.get(true) orelse return error.QueueEmpty;
    try testing.expect(@intFromPtr(queued_batch) == @intFromPtr(old_batch));
    queued_batch.deinit();
}

test "Events.getNewBatch with null prev_events" {
    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    const old_batch = events.batch;
    const new_batch = try events.getNewBatch(null);

    try testing.expect(@intFromPtr(new_batch) != @intFromPtr(old_batch));
    try testing.expect(@intFromPtr(events.batch) == @intFromPtr(new_batch));
    try testing.expect(new_batch.prev_events == null);

    // Clean up queued batch
    const queued_batch = queue.get(true) orelse return error.QueueEmpty;
    queued_batch.deinit();
}

test "Events.appendNewEvent with empty batch" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    try cl.event.setUserEventStatus(cl_event, .complete);

    var test_data: u32 = 42;
    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                const ptr: *u32 = @ptrCast(@alignCast(data.?));
                ptr.* = 100;
            }
        }.testCallback,
        .data = &test_data,
    };

    try cl.event.retain(cl_event);
    const result_set = events.appendNewEvent(.read, null, cl_event, callback) catch |err| {
        cl.event.release(cl_event);
        return err;
    };

    try testing.expect(events.batch.number_of_sets == 0);
    try testing.expect(result_set.operation == .read);
    try testing.expect(result_set.events_count == 1);
    try testing.expect(result_set.callbacks.items.len == 1);
}

test "Events.appendNewEvent with full batch creates new batch" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // Fill the batch to capacity
    events.batch.number_of_sets = BatchLength;

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    try cl.event.setUserEventStatus(cl_event, .complete);

    const prev_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(prev_event);

    try cl.event.setUserEventStatus(prev_event, .complete);

    const prev_events = [_]cl.event.Event{prev_event};

    const old_batch = events.batch;

    try cl.event.retain(cl_event);
    const result_set = events.appendNewEvent(.read, &prev_events, cl_event, null) catch |err| {
        cl.event.release(cl_event);
        return err;
    };

    // Should have created a new batch
    try testing.expect(@intFromPtr(events.batch) != @intFromPtr(old_batch));
    try testing.expect(events.batch.number_of_sets == 0);
    try testing.expect(result_set.operation == .read);

    // Clean up queued batch
    const queued_batch = queue.get(true) orelse return error.QueueEmpty;
    queued_batch.deinit();
}

test "Events.appendNewEvent with set becoming full" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    try cl.event.setUserEventStatus(cl_event, .complete);

    // Use write operation which becomes full after one event
    try cl.event.retain(cl_event);
    const result_set = events.appendNewEvent(.write, null, cl_event, null) catch |err| {
        cl.event.release(cl_event);
        return err;
    };

    try testing.expect(events.batch.number_of_sets == 1);
    try testing.expect(result_set.operation == .write);
    try testing.expect(result_set.events_count == 1);
}

test "Events.appendNewEvent with set full requiring new set" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // First, fill the first set with a write operation
    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);

    try cl.event.setUserEventStatus(cl_event1, .complete);

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(testing.allocator, .write, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result1 == .success_and_full);
    events.batch.number_of_sets = 1;

    // Now try to append a different operation
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    try cl.event.setUserEventStatus(cl_event2, .complete);

    try cl.event.retain(cl_event2);
    const result_set = events.appendNewEvent(.read, null, cl_event2, null) catch |err| {
        cl.event.release(cl_event2);
        return err;
    };

    try testing.expect(events.batch.number_of_sets == 1);
    try testing.expect(result_set.operation == .read);
    try testing.expect(result_set.events_count == 1);
    try testing.expect(@intFromPtr(result_set) == @intFromPtr(&events.batch.sets[1]));
}

test "Events.appendNewEvent with set full and batch full creates new batch" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // Fill all sets in the batch with write operations (which become full immediately)
    var events_to_cleanup: std.ArrayList(cl.event.Event) = .empty;
    defer {
        for (events_to_cleanup.items) |e| {
            cl.event.release(e);
        }
        events_to_cleanup.deinit(testing.allocator);
    }

    for (0..BatchLength) |i| {
        const cl_event = try cl.event.createUserEvent(context.ctx);
        {
            errdefer cl.event.release(cl_event);
            try cl.event.setUserEventStatus(cl_event, .complete);
            try cl.event.retain(cl_event);

            try events_to_cleanup.append(testing.allocator, cl_event);
        }

        const result = events.batch.sets[i].append(testing.allocator, .write, cl_event, null) catch |err| {
            cl.event.release(cl_event);
            return err;
        };
        try testing.expect(result == .success_and_full);
    }
    events.batch.number_of_sets = BatchLength;

    // Now try to append another event - should create new batch
    const cl_event_new = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event_new);

    try cl.event.setUserEventStatus(cl_event_new, .complete);

    const prev_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(prev_event);

    try cl.event.setUserEventStatus(prev_event, .complete);

    const prev_events = [_]cl.event.Event{prev_event};

    const old_batch = events.batch;

    try cl.event.retain(cl_event_new);
    const result_set = events.appendNewEvent(.read, &prev_events, cl_event_new, null) catch |err| {
        cl.event.release(cl_event_new);
        return err;
    };

    // Should have created a new batch
    try testing.expect(@intFromPtr(events.batch) != @intFromPtr(old_batch));
    try testing.expect(events.batch.number_of_sets == 0);
    try testing.expect(result_set.operation == .read);

    // Clean up queued batch
    const queued_batch = queue.get(true) orelse return error.QueueEmpty;
    queued_batch.deinit();
}

test "Events.deinit waits for pending events and cleans up" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);

    // Add some events
    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    try cl.event.retain(cl_event);
    const result = events.batch.sets[0].append(testing.allocator, .read, cl_event, null) catch |err| {
        cl.event.release(cl_event);
        return err;
    };
    try testing.expect(result == .success);

    // Set event to complete so deinit doesn't block
    try cl.event.setUserEventStatus(cl_event, .complete);

    // This should wait for events and clean up
    events.deinit();
}

test "Events integration test - multiple operations and batch transitions" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    var events_to_cleanup: std.ArrayList(cl.event.Event) = .empty;
    defer {
        for (events_to_cleanup.items) |e| {
            cl.event.release(e);
        }
        events_to_cleanup.deinit(testing.allocator);
    }

    // Test sequence: read -> read -> write -> read (should create transitions)
    
    // First read
    const read_event1 = try cl.event.createUserEvent(context.ctx);
    {
        errdefer cl.event.release(read_event1);
        try cl.event.setUserEventStatus(read_event1, .complete);
        try cl.event.retain(read_event1);

        try events_to_cleanup.append(testing.allocator, read_event1);
    }

    var result_set = events.appendNewEvent(.read, null, read_event1, null) catch |err| {
        cl.event.release(read_event1);
        return err;
    };
    try testing.expect(result_set.operation == .read);
    try testing.expect(events.batch.number_of_sets == 0);

    // Second read (should append to same set)
    const read_event2 = try cl.event.createUserEvent(context.ctx);
    {
        errdefer cl.event.release(read_event2);
        try cl.event.setUserEventStatus(read_event2, .complete);
        try cl.event.retain(read_event2);

        try events_to_cleanup.append(testing.allocator, read_event2);
    }

    result_set = events.appendNewEvent(.read, null, read_event2, null) catch |err| {
        cl.event.release(read_event2);
        return err;
    };
    try testing.expect(result_set.operation == .read);
    try testing.expect(result_set.events_count == 2);

    // Write operation (should cause transition)
    const write_event = try cl.event.createUserEvent(context.ctx);
    {
        errdefer cl.event.release(write_event);

        try cl.event.setUserEventStatus(write_event, .complete);
        try events_to_cleanup.append(testing.allocator, write_event);
    }

    const prev_events = events.getPrevEvents(.write);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 2);

    try cl.event.retain(write_event);
    result_set = events.appendNewEvent(.write, prev_events, write_event, null) catch |err| {
        cl.event.release(write_event);
        return err;
    };
    try testing.expect(result_set.operation == .write);
    try testing.expect(events.batch.number_of_sets == 2);

    // Another read (should go to next set)
    const read_event3 = try cl.event.createUserEvent(context.ctx);
    {
        errdefer cl.event.release(read_event3);

        try cl.event.setUserEventStatus(read_event3, .complete);
        try cl.event.retain(read_event3);

        try events_to_cleanup.append(testing.allocator, read_event3);
    }

    result_set = events.appendNewEvent(.read, null, read_event3, null) catch |err| {
        cl.event.release(read_event3);
        return err;
    };
    try testing.expect(result_set.operation == .read);
    try testing.expect(@intFromPtr(result_set) == @intFromPtr(&events.batch.sets[2]));
}

test {
    _ = Set;
    _ = Batch;
}
