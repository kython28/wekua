const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");
const Event = @import("event.zig");

pub const Length = switch (builtin.mode) {
    .Debug => 4,
    .ReleaseSafe, .ReleaseFast => 128,
    .ReleaseSmall => 16,
};

allocator: std.mem.Allocator,
prev_events: ?[]cl.event.Event,

events: [Length]Event,
events_num: u8,

pub fn initValue(
    self: *Batch,
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
        e.initValues(index, allocator);
    }

    self.events_num = 0;
}

pub fn init(
    allocator: std.mem.Allocator,
    prev_events: ?[]const cl.event.Event,
) !*Batch {
    const batch = try allocator.create(Batch);
    errdefer allocator.destroy(batch);

    try batch.initValue(allocator, prev_events);
    return batch;
}

pub inline fn empty(self: *const Batch) bool {
    return (self.events_num == 0);
}

pub inline fn full(self: *const Batch) bool {
    return (self.events_num == Length);
}

pub inline fn getPrevEvents(self: *const Batch) ?[]const cl.event.Event {
    return self.prev_events;
}

pub inline fn clear(self: *Batch) void {
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

pub fn restart(self: *Batch, new_prev_events: ?[]const cl.event.Event) !void {
    if (self.prev_events) |prev_events| {
        for (prev_events) |e| {
            cl.event.release(e);
        }
        self.allocator.free(prev_events);
        self.prev_events = null;
    }

    if (new_prev_events) |pv| {
        if (pv.len > Event.MaxEventsPerSet * 2) return error.EventsArrayTooLong;

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

pub fn waitForPendingEvents(self: *Batch) void {
    var events_num = self.events_num;
    if (events_num == 0) {
        if (self.events[0].operation == .none) {
            return;
        }

        events_num += 1;
    } else if (events_num < Length) {
        if (self.events[events_num].operation != .none) {
            events_num += 1;
        }
    }
    self.events_num = events_num;

    for (self.events[0..events_num]) |*event| {
        event.waitForEvents() catch |err| {
            std.debug.panic(
                "Unexpected error while waiting for events finalization: {s}",
                .{@errorName(err)},
            );
        };
    }
}

pub fn deinit(self: *Batch) void {
    self.clear();
    self.allocator.destroy(self);
}

const Batch = @This();

// Tests
const testing = std.testing;
const core = @import("core");

test "Batch.init with null prev_events" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    try testing.expect(batch.prev_events == null);
    try testing.expect(batch.events_num == 0);
    try testing.expect(batch.empty());
    try testing.expect(!batch.full());

    // Check that all events are properly initialized
    for (batch.events, 0..) |*event, i| {
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
    defer cl.event.release(cl_event1);

    const prev_events = [_]cl.event.Event{ cl_event1, cl_event2 };

    const batch = try Batch.init(testing.allocator, &prev_events);
    defer batch.deinit();

    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 2);
    try testing.expect(batch.events_num == 0);
}

test "Batch.empty and full state management" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    // Initially empty
    try testing.expect(batch.empty());
    try testing.expect(!batch.full());

    // Simulate adding events
    batch.events_num = 1;
    try testing.expect(!batch.empty());
    try testing.expect(!batch.full());

    // Fill to capacity
    batch.events_num = Length;
    try testing.expect(!batch.empty());
    try testing.expect(batch.full());

    // Reset
    batch.events_num = 0;
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
    defer batch.deinit();

    const returned_events = batch.getPrevEvents();
    try testing.expect(returned_events != null);
    try testing.expect(returned_events.?.len == 2);
    try testing.expect(returned_events.?[0] == cl_event1);
    try testing.expect(returned_events.?[1] == cl_event2);
}

test "Batch.getPrevEvents returns null when no prev_events" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

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
    defer batch.deinit();

    // Add some events to simulate state
    const test_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(test_event);

    try cl.event.retain(test_event);

    const result = batch.events[0].append(.read, test_event, null) catch |err| {
        cl.event.release(test_event);
        return err;
    };

    try testing.expect(result == .success);
    batch.events_num = 1;

    // Verify state before clear
    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.events[0].events_count == 1);

    // Clear should release everything
    batch.clear();

    try testing.expect(batch.prev_events == null);
    // Note: We can't easily test that events are cleared since clear() calls event.clear()
    // which deinitializes the callbacks ArrayList
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
    defer batch.deinit();

    // Add some state
    batch.events_num = 2;

    try batch.restart(null);

    try testing.expect(batch.prev_events == null);
    try testing.expect(batch.events_num == 0);
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
    defer batch.deinit();

    // Add some state
    batch.events_num = 1;

    try batch.restart(&new_prev_events);

    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 2);
    try testing.expect(batch.events_num == 0);
}

test "Batch.restart with too many events returns error" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    // Create an array that's too large
    const too_many_events = try testing.allocator.alloc(
        cl.event.Event,
        Event.MaxEventsPerSet * 2 + 1,
    );
    defer testing.allocator.free(too_many_events);

    const result = batch.restart(too_many_events);
    try testing.expectError(error.EventsArrayTooLong, result);
}

test "Batch.waitForPendingEvents with no events" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    // Should not panic or error
    batch.waitForPendingEvents();

    try testing.expect(batch.events_num == 0);
}

test "Batch.waitForPendingEvents with events in first slot" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    try cl.event.retain(cl_event);

    // Add event to first slot but don't increment events_num
    const result = batch.events[0].append(.read, cl_event, null) catch |err| {
        cl.event.release(cl_event);
        return err;
    };
    try testing.expect(result == .success);

    // Set event to complete so waitForEvents doesn't block
    try cl.event.setUserEventStatus(cl_event, .complete);

    batch.waitForPendingEvents();

    try testing.expect(batch.events_num == 1);
}

test "Batch.waitForPendingEvents with events beyond current count" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    try cl.event.retain(cl_event1);

    // Add events
    var result = batch.events[0].append(.read, cl_event1, null) catch |err| {
        cl.event.release(cl_event1);
        return err;
    };
    try testing.expect(result == .success);
    batch.events_num = 1;

    try cl.event.retain(cl_event2);

    result = batch.events[1].append(.write, cl_event2, null) catch |err| {
        cl.event.release(cl_event2);
        return err;
    };
    try testing.expect(result == .success_and_full);

    // Set events to complete
    try cl.event.setUserEventStatus(cl_event1, .complete);
    try cl.event.setUserEventStatus(cl_event2, .complete);

    batch.waitForPendingEvents();

    try testing.expect(batch.events_num == 2);
}

test "Batch.waitForPendingEvents at full capacity" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    // Fill to capacity
    batch.events_num = Length;

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

        const result = batch.events[i].append(.read, cl_event, null) catch |err| {
            cl.event.release(cl_event);
            return err;
        };
        try testing.expect(result == .success or result == .success_and_full);

        try cl.event.setUserEventStatus(cl_event, .complete);
    }

    batch.waitForPendingEvents();

    try testing.expect(batch.events_num == Length);
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
    defer batch.deinit();

    // Add some state
    const test_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(test_event);

    try cl.event.retain(test_event);

    const result = batch.events[0].append(.read, test_event, null) catch |err| {
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
    defer batch.deinit();

    // First restart with events
    const prev_events1 = [_]cl.event.Event{cl_event1};
    try batch.restart(&prev_events1);
    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 1);

    // Second restart with different events
    const prev_events2 = [_]cl.event.Event{ cl_event1, cl_event2 };
    try batch.restart(&prev_events2);
    try testing.expect(batch.prev_events != null);
    try testing.expect(batch.prev_events.?.len == 2);

    // Third restart with null
    try batch.restart(null);
    try testing.expect(batch.prev_events == null);
}

test "Batch events array initialization and indexing" {
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit();

    // Verify all events are properly initialized with correct indices
    for (batch.events, 0..) |*event, expected_index| {
        try testing.expect(event.index == expected_index);
        try testing.expect(event.operation == .none);
        try testing.expect(event.events_count == 0);

        // Verify getParent works correctly
        const parent = event.getParent();
        try testing.expect(@intFromPtr(parent) == @intFromPtr(batch));
    }
}
