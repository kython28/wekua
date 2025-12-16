const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("command_queue.zig");

const EventsArray = std.ArrayList([]const cl.event.Event);

command_queue: *CommandQueue,
events_batches: EventsArray,

pub fn init(command_queue: *CommandQueue) error{OutOfMemory}!*Pipeline {
    const self = try command_queue.context.allocator.create(Pipeline);
    errdefer command_queue.context.allocator.destroy(self);

    self.command_queue = command_queue;
    self.events_batches = .empty;

    return self;
}

pub fn deinit(self: *Pipeline) void {
    const allocator = self.command_queue.context.allocator;
    self.events_batches.deinit(allocator);
    allocator.destroy(self);
}

pub fn prevEvents(self: *Pipeline) ?[]const cl.event.Event {
    const events_batches = self.events_batches.items;
    if (events_batches.len == 0) return null;

    return events_batches[events_batches.len - 1];
}

pub fn append(self: *Pipeline, events: []const cl.event.Event) error{OutOfMemory}!void {
    const allocator = self.command_queue.context.allocator;

    const new_events = try allocator.dupe(cl.event.Event, events);
    errdefer allocator.free(new_events);

    try self.events_batches.append(allocator, new_events);
}

pub fn waitAndCleanup(self: *Pipeline) void {
    const events_batches = self.events_batches.items;
    for (events_batches) |events| {
        try cl.event.waitForMany(events);
    }

    for (events_batches) |events| {
        for (events) |event| {
            cl.event.release(event);
        }
    }

    self.events_batches.clearRetainingCapacity();
}


const Pipeline = @This();
