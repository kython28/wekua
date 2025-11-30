const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");
const Sets = @import("set.zig");

pub const Errors = cl.errors.OpenCLError || error{OutOfMemory};

pub const Length = switch (builtin.mode) {
    .Debug => 4,
    .ReleaseSafe, .ReleaseFast => 128,
    .ReleaseSmall => 16,
};

const utils = @import("utils");
pub const BatchQueue = utils.linked_list_module.LinkedList(Batch);
const BatchNode = BatchQueue.Node;

prev_events: ?[]cl.event.Event,

sets: [Length]Sets,
number_of_sets: u8,

pub fn initValues(
    self: *Batch,
    allocator: std.mem.Allocator,
    prev_events: ?[]const cl.event.Event,
) Errors!void {
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

    for (&self.sets, 0..) |*e, index| {
        e.initValues(index);
    }

    self.number_of_sets = 0;
}

pub fn init(
    allocator: std.mem.Allocator,
    prev_events: ?[]const cl.event.Event,
) Errors!*Batch {
    const node = try BatchNode.init(allocator, undefined);
    errdefer node.deinit(allocator);

    const batch = &node.data;
    try batch.initValues(allocator, prev_events);
    return batch;
}

pub inline fn empty(self: *const Batch) bool {
    return (self.number_of_sets == 0);
}

pub inline fn full(self: *const Batch) bool {
    return (self.number_of_sets == Length);
}

pub inline fn getPrevEvents(self: *const Batch) ?[]const cl.event.Event {
    return self.prev_events;
}

pub fn clear(self: *Batch, allocator: std.mem.Allocator) void {
    if (self.prev_events) |prev_events| {
        for (prev_events) |e| {
            cl.event.release(e);
        }
        allocator.free(prev_events);
    }

    for (self.sets[0..self.number_of_sets]) |*event| {
        event.clear(allocator);
    }

    self.number_of_sets = 0;
}

pub fn waitForPendingEvents(self: *Batch) void {
    var events_num = self.number_of_sets;
    if (events_num == 0) {
        if (self.sets[0].operation == .none) {
            return;
        }

        events_num += 1;
    } else if (events_num < Length) {
        if (self.sets[events_num].operation != .none) {
            events_num += 1;
        }
    }
    self.number_of_sets = events_num;

    for (self.sets[0..events_num]) |*event| {
        event.waitForEvents() catch |err| {
            std.debug.panic(
                "Unexpected error while waiting for events finalization: {s}",
                .{@errorName(err)},
            );
        };
    }
}

pub fn deinit(self: *Batch, allocator: std.mem.Allocator) void {
    self.clear(allocator);

    const node: *BatchNode = @fieldParentPtr("data", self);
    node.deinit(allocator);
}

pub inline fn push(self: *Batch, batches: *BatchQueue) void {
    const node: *BatchNode = @fieldParentPtr("data", self);
    batches.append_node(node);
}

const Batch = @This();

// Tests
const testing = std.testing;
const core = @import("core");

test "Batch.init with null prev_events" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    try testing.expect(batch.prev_events == null);
    try testing.expect(batch.number_of_sets == 0);
    try testing.expect(batch.empty());
    try testing.expect(!batch.full());

    // Check that all events are properly initialized
    for (&batch.sets, 0..) |*event, i| {
        try testing.expect(event.index == i);
        try testing.expect(event.operation == .none);
        try testing.expect(event.events_count == 0);
    }
}

test "Batch.init with prev_events" {
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

    const prev_events = [_]cl.event.Event{ cl_event1, cl_event2 };

    const batch = try Batch.init(testing.allocator, &prev_events);
    defer batch.deinit(testing.allocator);

    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 2);
    try testing.expect(batch.number_of_sets == 0);
}

test "Batch.empty and full state management" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    // Initially empty
    try testing.expect(batch.empty());
    try testing.expect(!batch.full());

    // Simulate adding events
    batch.number_of_sets = 1;
    try testing.expect(!batch.empty());
    try testing.expect(!batch.full());

    // Fill to capacity
    batch.number_of_sets = Length;
    try testing.expect(!batch.empty());
    try testing.expect(batch.full());

    // Reset
    batch.number_of_sets = 0;
    try testing.expect(batch.empty());
    try testing.expect(!batch.full());
}

test "Batch.getPrevEvents returns correct events" {
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

    const prev_events = [_]cl.event.Event{ cl_event1, cl_event2 };

    const batch = try Batch.init(testing.allocator, &prev_events);
    defer batch.deinit(testing.allocator);

    const returned_events = batch.getPrevEvents();
    try testing.expect(returned_events != null);
    try testing.expect(returned_events.?.len == 2);
    try testing.expect(returned_events.?[0] == cl_event1);
    try testing.expect(returned_events.?[1] == cl_event2);
}

test "Batch.getPrevEvents returns null when no prev_events" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    const returned_events = batch.getPrevEvents();
    try testing.expect(returned_events == null);
}

test "Batch.clear releases prev_events and clears event states" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const prev_events = [_]cl.event.Event{cl_event};

    const batch = try Batch.init(testing.allocator, &prev_events);
    defer batch.deinit(testing.allocator);

    // Add some events to simulate state
    const test_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(test_event);

    try cl.event.retain(test_event);

    const result = batch.sets[0].append(testing.allocator, .read, test_event, null) catch |err| {
        cl.event.release(test_event);
        return err;
    };

    try testing.expect(result == .success);
    batch.number_of_sets = 1;

    // Verify state before clear
    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.sets[0].events_count == 1);

    // Clear should release everything
    batch.clear(testing.allocator);

    try testing.expect(batch.prev_events == null);
}

test "Batch.restart with null new_prev_events" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const prev_events = [_]cl.event.Event{cl_event};

    const batch = try Batch.init(testing.allocator, &prev_events);
    defer batch.deinit(testing.allocator);

    // Add some state
    batch.number_of_sets = 2;

    try batch.restart(testing.allocator, null);

    try testing.expect(batch.prev_events == null);
    try testing.expect(batch.number_of_sets == 0);
}

test "Batch.restart with new prev_events" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const old_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(old_event);
    const new_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(new_event1);
    const new_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(new_event2);

    const old_prev_events = [_]cl.event.Event{old_event};
    const new_prev_events = [_]cl.event.Event{ new_event1, new_event2 };

    const batch = try Batch.init(testing.allocator, &old_prev_events);
    defer batch.deinit(testing.allocator);

    // Add some state
    batch.number_of_sets = 1;

    try batch.restart(testing.allocator, &new_prev_events);

    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 2);
    try testing.expect(batch.number_of_sets == 0);
}

test "Batch.waitForPendingEvents with no events" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    // Should not panic or error
    batch.waitForPendingEvents();

    try testing.expect(batch.number_of_sets == 0);
}

test "Batch.waitForPendingEvents with events in first slot" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    try cl.event.retain(cl_event);

    // Add event to first slot but don't increment events_num
    const result = batch.sets[0].append(testing.allocator, .read, cl_event, null) catch |err| {
        cl.event.release(cl_event);
        return err;
    };
    try testing.expect(result == .success);

    // Set event to complete so waitForEvents doesn't block
    try cl.event.setUserEventStatus(cl_event, .complete);

    batch.waitForPendingEvents();

    try testing.expect(batch.number_of_sets == 1);
}

test "Batch.waitForPendingEvents with events beyond current count" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    try cl.event.retain(cl_event1);

    // Add events
    var result = batch.sets[0].append(testing.allocator, .read, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result == .success);
    batch.number_of_sets = 1;

    try cl.event.retain(cl_event2);

    result = batch.sets[1].append(testing.allocator, .write, cl_event2, null) catch |err| {
        cl.event.release(cl_event2);
        return err;
    };
    try testing.expect(result == .success_and_full);

    // Set events to complete
    try cl.event.setUserEventStatus(cl_event1, .complete);
    try cl.event.setUserEventStatus(cl_event2, .complete);

    batch.waitForPendingEvents();

    try testing.expect(batch.number_of_sets == 2);
}

test "Batch.waitForPendingEvents at full capacity" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    // Fill to capacity
    batch.number_of_sets = Length;

    const events_to_cleanup = try testing.allocator.alloc(cl.event.Event, Length);
    var number_of_events_to_cleanup: usize = 0;
    defer {
        for (events_to_cleanup[0..number_of_events_to_cleanup]) |e| {
            cl.event.release(e);
        }
        testing.allocator.free(events_to_cleanup);
    }

    // Add events to each slot
    for (0..Length) |i| {
        const cl_event = try cl.event.createUserEvent(context.ctx);
        events_to_cleanup[i] = cl_event;
        number_of_events_to_cleanup += 1;

        const result = batch.sets[i].append(testing.allocator, .read, cl_event, null) catch |err| {
            cl.event.release(cl_event);
            return err;
        };
        try testing.expect(result == .success or result == .success_and_full);

        try cl.event.setUserEventStatus(cl_event, .complete);
    }
    number_of_events_to_cleanup = 0;

    batch.waitForPendingEvents();

    try testing.expect(batch.number_of_sets == Length);
}

test "Batch.deinit calls clear and destroys" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const prev_events = [_]cl.event.Event{cl_event};

    const batch = try Batch.init(testing.allocator, &prev_events);
    defer batch.deinit(testing.allocator);

    // Add some state
    const test_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(test_event);

    try cl.event.retain(test_event);

    const result = batch.sets[0].append(testing.allocator, .read, test_event, null) catch |err| {
        cl.event.release(test_event);
        return err;
    };
    try testing.expect(result == .success);

    // Set event to complete
    try cl.event.setUserEventStatus(test_event, .complete);
}

test "Batch multiple restart cycles" {
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

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    // First restart with events
    const prev_events1 = [_]cl.event.Event{cl_event1};
    try batch.restart(testing.allocator, &prev_events1);
    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 1);

    // Second restart with different events
    const prev_events2 = [_]cl.event.Event{ cl_event1, cl_event2 };
    try batch.restart(testing.allocator, &prev_events2);
    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 2);

    // Third restart with null
    try batch.restart(testing.allocator, null);
    try testing.expect(batch.prev_events == null);
}

test "Batch events array initialization and indexing" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    // Verify all events are properly initialized with correct indices
    for (&batch.sets, 0..) |*event, expected_index| {
        try testing.expect(event.index == expected_index);
        try testing.expect(event.operation == .none);
        try testing.expect(event.events_count == 0);

        // Verify getParent works correctly
        const parent = event.getParent();
        try testing.expect(@intFromPtr(parent) == @intFromPtr(batch));
    }
}
