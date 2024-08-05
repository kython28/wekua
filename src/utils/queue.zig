const std = @import("std");

const wLinkedList = @import("linked_list.zig");
// const wLinkedList = linked_list.wLinkedList;

pub const wQueue = struct {
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,
    queue: wLinkedList,

    releasing: bool,

    pub fn init(allocator: std.mem.Allocator) wQueue {
        return wQueue{
            .mutex = std.Thread.Mutex{},
            .queue = wLinkedList.init(allocator),
            .cond = std.Thread.Condition{},
            .releasing = false
        };
    }

    pub fn put(self: *wQueue, data: ?*anyopaque) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.queue.append(data);
        self.cond.signal();
    }

    pub fn get(self: *wQueue, wait: bool) !?*anyopaque {
        self.mutex.lock();
        defer self.mutex.unlock();
        defer self.cond.signal();

        if (!wait and self.queue.last == null) return null;

        while (self.queue.last == null) {
            if (self.releasing) return null;

            self.cond.wait(&self.mutex);
        }

        const data = try self.queue.pop();
        return data;
    }

    pub fn release(self: *wQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.releasing = true;
        self.cond.broadcast();

        while (self.queue.last != null) {
            self.cond.wait(&self.mutex);
        }
    }
};
