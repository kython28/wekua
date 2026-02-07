const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("command_queue.zig");

command_queue: *CommandQueue,
allocator: std.mem.Allocator,
events: std.ArrayList(cl.event.Event),
prev_batch_start: usize,

pub fn init(command_queue: *CommandQueue) error{OutOfMemory}!*Pipeline {
    const allocator = command_queue.context.allocator;

    const self = try allocator.create(Pipeline);
    self.* = .{
        .allocator = allocator,
        .command_queue = command_queue,
        .events = .empty,
        .prev_batch_start = 0,
    };

    return self;
}

pub fn deinit(self: *Pipeline) void {
    const allocator = self.allocator;
    self.events.deinit(allocator);
    allocator.destroy(self);
}

pub fn prealloc(self: *Pipeline, capacity: usize) error{OutOfMemory}!void {
    try self.events.ensureTotalCapacity(self.allocator, capacity);
}

pub fn prevEvents(self: *Pipeline) ?[]const cl.event.Event {
    const items = self.events.items;
    if (items.len == 0) return null;

    return items[self.prev_batch_start..];
}

pub fn append(self: *Pipeline, events: []const cl.event.Event) error{OutOfMemory}!void {
    self.prev_batch_start = self.events.items.len;
    try self.events.appendSlice(self.allocator, events);
}

pub fn waitAndCleanup(self: *Pipeline) void {
    const items = self.events.items;
    if (items.len == 0) return;

    cl.event.waitForMany(items) catch |err| {
        std.debug.panic("Unexpected error ({s}) while waiting for events", .{@errorName(err)});
    };

    for (items) |event| {
        cl.event.release(event);
    }

    self.events.clearRetainingCapacity();
    self.prev_batch_start = 0;
}

pub fn clear(self: *Pipeline) void {
    self.events.clearAndFree(self.allocator);
    self.prev_batch_start = 0;
}


const Pipeline = @This();

// Unit Tests
const testing = std.testing;
const core = @import("main.zig");

test "Pipeline.init - basic initialization" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    try testing.expect(pipeline.command_queue == command_queue);
    try testing.expectEqual(@as(usize, 0), pipeline.events.items.len);
}

test "Pipeline.prevEvents - returns null when empty" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const prev_events = pipeline.prevEvents();
    try testing.expect(prev_events == null);
}

test "Pipeline.append - adds events to pipeline" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    // Create test events
    const event1 = try cl.event.createUserEvent(context.cl_context);
    defer cl.event.release(event1);
    const event2 = try cl.event.createUserEvent(context.cl_context);
    defer cl.event.release(event2);

    const events = [_]cl.event.Event{ event1, event2 };

    // Append events
    try pipeline.append(&events);

    try testing.expectEqual(@as(usize, 2), pipeline.events.items.len);
}

test "Pipeline.prevEvents - returns last appended events" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    // Create first batch of events
    const event1 = try cl.event.createUserEvent(context.cl_context);
    defer cl.event.release(event1);
    const event2 = try cl.event.createUserEvent(context.cl_context);
    defer cl.event.release(event2);

    const events1 = [_]cl.event.Event{ event1, event2 };
    try pipeline.append(&events1);

    // Create second batch of events
    const event3 = try cl.event.createUserEvent(context.cl_context);
    defer cl.event.release(event3);

    const events2 = [_]cl.event.Event{event3};
    try pipeline.append(&events2);

    // prevEvents should return only the last batch
    const prev_events = pipeline.prevEvents();
    try testing.expect(prev_events != null);
    try testing.expectEqual(@as(usize, 1), prev_events.?.len);
    try testing.expectEqual(event3, prev_events.?[0]);
}

test "Pipeline.append - multiple batches" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    // Append multiple batches
    for (0..5) |i| {
        const event = try cl.event.createUserEvent(context.cl_context);
        defer cl.event.release(event);

        const events = [_]cl.event.Event{event};
        try pipeline.append(&events);

        try testing.expectEqual(i + 1, pipeline.events.items.len);
    }
}

test "Pipeline.waitAndCleanup - clears all events" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    // Create and append events
    const event1 = try cl.event.createUserEvent(context.cl_context);
    const event2 = try cl.event.createUserEvent(context.cl_context);

    const events = [_]cl.event.Event{ event1, event2 };
    try pipeline.append(&events);

    try testing.expectEqual(@as(usize, 2), pipeline.events.items.len);

    // Mark events as complete
    try cl.event.setUserEventStatus(event1, .complete);
    try cl.event.setUserEventStatus(event2, .complete);

    // Wait and cleanup
    pipeline.waitAndCleanup();

    // After cleanup, events should be cleared but capacity retained
    try testing.expectEqual(@as(usize, 0), pipeline.events.items.len);
    try testing.expect(pipeline.events.capacity > 0);

    // prevEvents should return null after cleanup
    const prev_events = pipeline.prevEvents();
    try testing.expect(prev_events == null);
}

test "Pipeline.clear - frees all memory" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    // Create and append events
    const event1 = try cl.event.createUserEvent(context.cl_context);
    defer cl.event.release(event1);

    const events = [_]cl.event.Event{event1};
    try pipeline.append(&events);

    try testing.expect(pipeline.events.capacity > 0);

    // Clear should free all memory
    pipeline.clear();

    try testing.expectEqual(@as(usize, 0), pipeline.events.items.len);
    try testing.expectEqual(@as(usize, 0), pipeline.events.capacity);
    try testing.expectEqual(@as(usize, 0), pipeline.prev_batch_start);

    // prevEvents should return null
    try testing.expect(pipeline.prevEvents() == null);
}

test "Pipeline.deinit - proper cleanup" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);

    // Add some events
    const event = try cl.event.createUserEvent(context.cl_context);
    const events = [_]cl.event.Event{event};
    try pipeline.append(&events);

    // Mark event as complete before cleanup
    try cl.event.setUserEventStatus(event, .complete);

    // deinit should not leak memory
    pipeline.deinit();
}

test "Pipeline.append - empty events array" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const empty_events: []const cl.event.Event = &.{};
    try pipeline.append(empty_events);

    // Flat list has 0 items, but prev_batch_start was updated
    try testing.expectEqual(@as(usize, 0), pipeline.events.items.len);
    try testing.expectEqual(@as(usize, 0), pipeline.prev_batch_start);
}

test "Pipeline - multiple pipelines on same command queue" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const pipeline1 = try Pipeline.init(command_queue);
    defer pipeline1.deinit();

    const pipeline2 = try Pipeline.init(command_queue);
    defer pipeline2.deinit();

    // Both should reference the same command queue
    try testing.expect(pipeline1.command_queue == pipeline2.command_queue);

    // But should have independent events
    const event1 = try cl.event.createUserEvent(context.cl_context);
    defer cl.event.release(event1);

    const events1 = [_]cl.event.Event{event1};
    try pipeline1.append(&events1);

    try testing.expectEqual(@as(usize, 1), pipeline1.events.items.len);
    try testing.expectEqual(@as(usize, 0), pipeline2.events.items.len);
}
