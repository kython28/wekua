const std = @import("std");

const cl = @import("opencl");

const Event = @import("event.zig");
const Batch = @import("batch.zig");
pub const Set = @import("set.zig");

const Operation = Event.Operation;
const BatchLength = Batch.Length;

const queue_module = @import("utils").queue_module;
pub const BatchQueue = queue_module.Queue(*Batch);
pub const UserCallback = Event.UserCallback;

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

    const events_num = batch.events_num;
    if (events_num == BatchLength) {
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
) !*Event {
    var batch = self.batch;

    var events_num = batch.events_num;
    if (events_num == BatchLength) {
        batch = try self.getNewBatch(prev_Events);
        events_num = 0;
    }

    var event: *Event = &batch.events[events_num];
    loop: switch (try event.append(new_op, new_Event, user_callback)) {
        .success => {},
        .full => {
            events_num += 1;
            if (events_num == BatchLength) {
                batch.events_num = BatchLength;

                batch = try self.getNewBatch(prev_Events);
                events_num = 0;
            }

            event = &batch.events[events_num];
            const new_result = try event.append(new_op, new_Event, user_callback);
            continue :loop new_result;
        },
        .success_and_full => {
            events_num += 1;
        },
    }

    batch.events_num = events_num;
    return event;
}

const Events = @This();

test {
    std.testing.refAllDecls(Event);
}
