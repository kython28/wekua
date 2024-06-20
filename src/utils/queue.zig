const std = @import("std");

const wMutex = @import("mutex.zig").wMutex;
const wCondition = @import("condition.zig").wCondition;
const linked_list = @import("linked_list.zig");
const wLinkedList = linked_list.wLinkedList;

pub const wQueue = struct {
    mutex: wMutex,
    cond: wCondition,
    queue: wLinkedList,

    pub fn init(allocator: *const std.mem.Allocator) !wQueue {
        return wQueue{
            .mutex = wMutex{},
            .queue = try linked_list.create(allocator),
            .cond = wCondition{}
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
