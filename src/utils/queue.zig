const std = @import("std");

const linked_list = @import("linked_list.zig");
const wLinkedList = linked_list.wLinkedList;

pub const wQueue = struct {
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,
    queue: wLinkedList,

    pub fn init(allocator: std.mem.Allocator) !wQueue {
        return wQueue{
            .mutex = std.Thread.Mutex{},
            .queue = try linked_list.create(allocator),
            .cond = std.Thread.Condition{}
        };
    }

    pub fn put(self: *wQueue, data: ?*anyopaque) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try linked_list.append(self.queue, data);
        self.cond.signal();
    }

    pub fn get(self: *wQueue, wait: bool) !?*anyopaque {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!wait and self.queue.last == null) return null;

        while (self.queue.last == null) {
            self.cond.wait(&self.mutex);
        }

        const data = try linked_list.pop(self.queue);
        return data;
    }

    pub fn join(self: *wQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.queue.last != null) {
            self.cond.wait(&self.mutex);
        }
    }

    pub fn release(self: *wQueue) void {
        // self.join();
        linked_list.release(self.queue) catch unreachable;
    }
};
