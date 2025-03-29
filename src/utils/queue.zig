const std = @import("std");

const linked_list = @import("linked_list.zig");

pub fn Queue(comptime T: type) type {
    const LinkedList = linked_list.LinkedList(T);
    return struct {
        mutex: std.Thread.Mutex,
        cond: std.Thread.Condition,
        list: LinkedList,

        releasing: bool,

        const this = @This();

        pub fn init(allocator: std.mem.Allocator) this {
            return .{
                .mutex = std.Thread.Mutex{},
                .list = LinkedList.init(allocator),
                .cond = std.Thread.Condition{},
                .releasing = false
            };
        }

        pub fn put(self: *this, data: T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            try self.list.append(data);
            self.cond.signal();
        }

        pub fn put_node(self: *this, node: LinkedList.Node) void {
            const mutex = &self.mutex;
            mutex.lock();
            defer mutex.unlock();

            self.list.append_node(node);
            self.cond.signal();
        }

        pub fn get_node(self: *this, wait: bool) !?LinkedList.Node {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.first == null) return null;

            while (self.list.first == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const last_node = try self.list.popleft_node();
            return last_node;
        }

        pub fn get(self: *this, wait: bool) !?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.first == null) return null;

            while (self.list.first == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const data = try self.list.popleft();
            return data;
        }

        pub fn get_last_node(self: *this, wait: bool) !?LinkedList.Node {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.last == null) return null;

            while (self.list.last == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const last_node = try self.list.pop_node();
            return last_node;
        }

        pub fn get_last(self: *this, wait: bool) !?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            defer self.cond.signal();

            if (!wait and self.list.last == null) return null;

            while (self.list.last == null) {
                if (self.releasing) return null;

                self.cond.wait(&self.mutex);
            }

            const data = try self.list.pop();
            return data;
        }

        pub fn release(self: *this) void {
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
