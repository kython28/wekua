const std = @import("std");

const linked_list = @import("linked_list.zig");

pub fn Queue(comptime T: type) type {
    const LinkedList = linked_list.LinkedList(T);
    return struct {
        mutex: std.Thread.Mutex,
        cond: std.Thread.Condition,
        list: LinkedList,

        releasing: bool,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .mutex = std.Thread.Mutex{},
                .list = LinkedList.init(allocator),
                .cond = std.Thread.Condition{},
                .releasing = false
            };
        }

        pub fn isEmpty(self: *Self) bool {
            self.mutex.lock();
            defer self.mutex.unlock();

            return self.list.isEmpty();
        }

        pub fn put(self: *Self, data: T) error{OutOfMemory}!void {
            self.mutex.lock();
            defer self.mutex.unlock();

            try self.list.append(data);
            self.cond.signal();
        }

        pub fn putNode(self: *Self, node: LinkedList.Node) void {
            const mutex = &self.mutex;
            mutex.lock();
            defer mutex.unlock();

            self.list.appendNode(node);
            self.cond.signal();
        }

        pub fn getNode(self: *Self, wait: bool) ?LinkedList.Node {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.first == null) return null;

            while (self.list.first == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const last_node = self.list.popLeftNode();
            return last_node;
        }

        pub fn get(self: *Self, wait: bool) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.first == null) return null;

            while (self.list.first == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const data = self.list.popLeft();
            return data;
        }

        pub fn getLastNode(self: *Self, wait: bool) ?LinkedList.Node {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.last == null) return null;

            while (self.list.last == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const last_node = self.list.popNode();
            return last_node;
        }

        pub fn getLast(self: *Self, wait: bool) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.last == null) return null;

            while (self.list.last == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const data = self.list.pop();
            return data;
        }

        pub fn deinit(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.releasing = true;
            self.cond.broadcast();

            while (self.list.last != null) {
                self.cond.wait(&self.mutex);
            }
        }
    };
}
