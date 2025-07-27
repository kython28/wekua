const std = @import("std");

const cl = @import("opencl");

const Set = @import("set.zig");
const Batch = @import("batch.zig");
pub const Combined = @import("combined.zig");

const Operation = Set.Operation;
const BatchLength = Batch.Length;

const queue_module = @import("utils").queue_module;
pub const BatchQueue = queue_module.Queue(*Batch);
pub const UserCallback = Set.UserCallback;

allocator: std.mem.Allocator,
batch: *Batch,
events_releaser_queue: *BatchQueue,

pub fn init(
    self: *Events,
    allocator: std.mem.Allocator,
    queue: *BatchQueue,
) !void {
    self.allocator = allocator;

    const batch = try Batch.init(allocator, null);
    errdefer batch.deinit();

    self.batch = batch;
    self.events_releaser_queue = queue;
}

pub fn deinit(self: *Events) void {
    self.batch.waitForPendingEvents();
    self.batch.deinit();
}

pub fn getPrevEvents(self: *Events, new_op: Operation) ?[]const cl.event.Event {
    if (new_op == .none) unreachable;

    const batch = self.batch;

    const events_num = batch.number_of_sets;
    if (events_num == BatchLength) {
        const event: *Set = &batch.sets[events_num - 1];
        return event.toSlice();
    }

    const event: *Set = &batch.sets[events_num];
    switch (event.operation) {
        .read => switch (new_op) {
            .write, .partial_write => {
                batch.number_of_sets += 1;
                return event.toSlice();
            },
            .read => {},
            else => unreachable,
        },
        .write => unreachable,
        .partial_write => {
            batch.number_of_sets += 1;
            return event.toSlice();
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
) !?[]cl.event.Event {
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


fn getNewBatch(self: *Events, prev_events: ?[]const cl.event.Event) !*Batch {
    const old_batch = self.batch;

    const allocator = self.allocator;
    const new_batch = try Batch.init(allocator, prev_events);
    errdefer new_batch.deinit();

    self.batch = new_batch;
    errdefer self.batch = old_batch;

    try self.events_releaser_queue.put(old_batch);
    return new_batch;
}

pub fn appendNewEvent(
    self: *Events,
    new_op: Operation,
    prev_Events: ?[]const cl.event.Event,
    new_Event: cl.event.Event,
    user_callback: ?UserCallback,
) !*Set {
    var batch = self.batch;

    var events_num = batch.number_of_sets;
    if (events_num == BatchLength) {
        batch = try self.getNewBatch(prev_Events);
        events_num = 0;
    }

    var event: *Set = &batch.sets[events_num];
    loop: switch (try event.append(new_op, new_Event, user_callback)) {
        .success => {},
        .full => {
            events_num += 1;
            if (events_num == BatchLength) {
                batch.number_of_sets = BatchLength;

                batch = try self.getNewBatch(prev_Events);
                events_num = 0;
            }

            event = &batch.sets[events_num];
            const new_result = try event.append(new_op, new_Event, user_callback);
            continue :loop new_result;
        },
        .success_and_full => {
            events_num += 1;
        },
    }

    batch.number_of_sets = events_num;
    return event;
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

    try testing.expect(events.allocator.ptr == testing.allocator.ptr);
    try testing.expect(events.batch.number_of_sets == 0);
    try testing.expect(@intFromPtr(events.events_releaser_queue) == @intFromPtr(&queue));
}

test "Events.getPrevEvents with none operation panics" {
    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    // This should panic, but we can't easily test panics in Zig
    // Just document the expected behavior
}

test "Events.getPrevEvents with empty batch and no prev events" {
    var queue = BatchQueue.init(testing.allocator);
    defer queue.deinit();

    var events: Events = undefined;
    try events.init(testing.allocator, &queue);
    defer events.deinit();

    const result = events.getPrevEvents(.read);
    try testing.expect(result == null);
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

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(.read, cl_event1, null) catch |err| {
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

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(.read, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result1 == .success);

    // Now getPrevEvents with write should increment and return events
    const prev_events = events.getPrevEvents(.write);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 1);
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

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(.partial_write, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result1 == .success_and_full);

    // Any new operation should increment and return events
    const prev_events = events.getPrevEvents(.read);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 1);
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

    try cl.event.retain(cl_event);
    const result = events.batch.sets[BatchLength - 1].append(.read, cl_event, null) catch |err| {
        cl.event.release(cl_event);
        return err;
    };
    try testing.expect(result == .success);

    const prev_events = events.getPrevEvents(.write);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 1);
}

test "Events.getPrevEvents returns previous event when events_num > 0" {
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

    // Add events to create a scenario with events_num > 0
    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    try cl.event.retain(cl_event1);
    var result = events.batch.sets[0].append(.write, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result == .success_and_full);
    events.batch.number_of_sets = 1;

    try cl.event.retain(cl_event2);
    result = events.batch.sets[1].append(.read, cl_event2, null) catch |err| {
        cl.event.release(cl_event2);
        return err;
    };
    try testing.expect(result == .success);

    // Now getPrevEvents should return the previous event (index 0)
    const prev_events = events.getPrevEvents(.write);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 1);
    try testing.expect(prev_events.?[0] == cl_event1);
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
    const queued_batch = try queue.get();
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
    const queued_batch = try queue.get();
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

    const result_set = try events.appendNewEvent(.read, null, cl_event, callback);

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

    const prev_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(prev_event);

    const prev_events = [_]cl.event.Event{prev_event};

    const old_batch = events.batch;
    const result_set = try events.appendNewEvent(.read, &prev_events, cl_event, null);

    // Should have created a new batch
    try testing.expect(@intFromPtr(events.batch) != @intFromPtr(old_batch));
    try testing.expect(events.batch.number_of_sets == 0);
    try testing.expect(result_set.operation == .read);

    // Clean up queued batch
    const queued_batch = try queue.get();
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

    // Use write operation which becomes full after one event
    const result_set = try events.appendNewEvent(.write, null, cl_event, null);

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

    try cl.event.retain(cl_event1);
    const result1 = events.batch.sets[0].append(.write, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result1 == .success_and_full);
    events.batch.number_of_sets = 1;

    // Now try to append a different operation
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    const result_set = try events.appendNewEvent(.read, null, cl_event2, null);

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
    var events_to_cleanup = std.ArrayList(cl.event.Event).init(testing.allocator);
    defer {
        for (events_to_cleanup.items) |e| {
            cl.event.release(e);
        }
        events_to_cleanup.deinit();
    }

    for (0..BatchLength) |i| {
        const cl_event = try cl.event.createUserEvent(context.ctx);
        try events_to_cleanup.append(cl_event);

        try cl.event.retain(cl_event);
        const result = events.batch.sets[i].append(.write, cl_event, null) catch |err| {
            cl.event.release(cl_event);
            return err;
        };
        try testing.expect(result == .success_and_full);
    }
    events.batch.number_of_sets = BatchLength;

    // Now try to append another event - should create new batch
    const cl_event_new = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event_new);

    const prev_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(prev_event);

    const prev_events = [_]cl.event.Event{prev_event};

    const old_batch = events.batch;
    const result_set = try events.appendNewEvent(.read, &prev_events, cl_event_new, null);

    // Should have created a new batch
    try testing.expect(@intFromPtr(events.batch) != @intFromPtr(old_batch));
    try testing.expect(events.batch.number_of_sets == 0);
    try testing.expect(result_set.operation == .read);

    // Clean up queued batch
    const queued_batch = try queue.get();
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
    const result = events.batch.sets[0].append(.read, cl_event, null) catch |err| {
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

    var events_to_cleanup = std.ArrayList(cl.event.Event).init(testing.allocator);
    defer {
        for (events_to_cleanup.items) |e| {
            cl.event.release(e);
        }
        events_to_cleanup.deinit();
    }

    // Test sequence: read -> read -> write -> read (should create transitions)
    
    // First read
    const read_event1 = try cl.event.createUserEvent(context.ctx);
    try events_to_cleanup.append(read_event1);
    try cl.event.setUserEventStatus(read_event1, .complete);

    var result_set = try events.appendNewEvent(.read, null, read_event1, null);
    try testing.expect(result_set.operation == .read);
    try testing.expect(events.batch.number_of_sets == 0);

    // Second read (should append to same set)
    const read_event2 = try cl.event.createUserEvent(context.ctx);
    try events_to_cleanup.append(read_event2);
    try cl.event.setUserEventStatus(read_event2, .complete);

    result_set = try events.appendNewEvent(.read, null, read_event2, null);
    try testing.expect(result_set.operation == .read);
    try testing.expect(result_set.events_count == 2);

    // Write operation (should cause transition)
    const write_event = try cl.event.createUserEvent(context.ctx);
    try events_to_cleanup.append(write_event);
    try cl.event.setUserEventStatus(write_event, .complete);

    const prev_events = events.getPrevEvents(.write);
    try testing.expect(prev_events != null);
    try testing.expect(prev_events.?.len == 2);

    result_set = try events.appendNewEvent(.write, prev_events, write_event, null);
    try testing.expect(result_set.operation == .write);
    try testing.expect(events.batch.number_of_sets == 1);

    // Another read (should go to next set)
    const read_event3 = try cl.event.createUserEvent(context.ctx);
    try events_to_cleanup.append(read_event3);
    try cl.event.setUserEventStatus(read_event3, .complete);

    result_set = try events.appendNewEvent(.read, null, read_event3, null);
    try testing.expect(result_set.operation == .read);
    try testing.expect(@intFromPtr(result_set) == @intFromPtr(&events.batch.sets[2]));
}

test {
    std.testing.refAllDecls(Set);
}
