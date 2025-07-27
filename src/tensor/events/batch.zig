const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");
const Event = @import("event.zig");

const Lenght = switch (builtin.mode) {
    .Debug => 4,
    .ReleaseSafe, .ReleaseFast => 128,
    .ReleaseSmall => 16,
};

allocator: std.mem.Allocator,
prev_events: ?[]cl.event.Event,

events: [Lenght]Event,
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
        e.init(index, allocator);
    }

    self.events_num = 0;
}

pub fn init(
    allocator: std.mem.Allocator,
    prev_events: ?[]const cl.event.Event,
) !Batch {
    const batch = try allocator.create(Batch);
    errdefer allocator.destroy(batch);

    try batch.initValue(allocator, prev_events);
    return batch;
}

pub inline fn empty(self: *const Batch) bool {
    return (self.events_num == 0);
}

pub inline fn full(self: *const Batch) bool {
    return (self.events_num == Lenght);
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
    } else if (events_num < Lenght) {
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
