const std = @import("std");

const EventManager = @import("../tensor/event_manager.zig");

const queue = @import("../utils/queue.zig");
pub const EventsBatchQueue = queue.Queue(*EventManager.EventsBatch);


pub fn eventsBatchReleaserWorker(events_batch_queue: *EventsBatchQueue) void {
    while (true) {
        const maybe_batch = events_batch_queue.get(true) catch |err| {
            std.debug.panic("Unexpected error while getting events batch: {s}", .{@errorName(err)});
        };

        const batch = maybe_batch orelse break;
        batch.waitForPendingEvents();
        batch.release();
    }
}
