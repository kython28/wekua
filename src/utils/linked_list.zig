const std = @import("std");
const builtin = @import("builtin");

pub fn LinkedList(comptime T: type) type {
    return struct {
        const LinkedListNode = struct {
            data: T,

            prev: ?*LinkedListNode,
            next: ?*LinkedListNode
        };

        pub const Node = *LinkedListNode;

        pub const Errors = error {
            CannotHasElementsWhenRealasing,
            LinkedListEmpty
        };

        allocator: std.mem.Allocator,
        first: ?Node = null,
        last: ?Node = null,
        len: usize = 0,

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .allocator = allocator,
            };
        }

        pub inline fn isEmpty(self: *const Self) bool {
            return (self.len == 0);
        }

        pub fn createNewNode(self: *const Self, data: T) error{OutOfMemory}!Node {
            const new_node = try self.allocator.create(LinkedListNode);
            new_node.data = data;
            return new_node;
        }

        pub fn releaseNode(self: *const Self, node: Node) void {
            self.allocator.destroy(node);
        }

        pub fn unlinkNode(self: *Self, node: Node) void {
            if (builtin.mode == .Debug and self.len == 0) {
                @panic("Trying to pop elements while linked list is empty");
            }

            const prev_node = node.prev;
            const next_node = node.next;

            if (prev_node) |n| {
                n.next = next_node;
            }else{
                self.first = next_node;
            }

            if (next_node) |n| {
                n.prev = prev_node;
            }else{
                self.last = prev_node;
            }

            self.len -= 1;
        }

        pub fn appendNode(self: *Self, new_node: Node) void {
            if (self.last) |last_node| {
                last_node.next = new_node;
            }

            new_node.prev = self.last;
            new_node.next = null;

            self.last = new_node;

            if (self.first == null) {
                self.first = new_node;
            }

            self.len += 1;
        }

        pub fn append(self: *Self, data: T) error{OutOfMemory}!void {
            const new_node = try self.createNewNode(data);
            self.appendNode(new_node);
        }

        pub fn appendLeftNode(self: *Self, new_node: Node) void {
            if (self.first) |first_node| {
                first_node.prev = new_node;
            }

            new_node.prev = null;
            new_node.next = self.first;

            self.first = new_node;
            if (self.last == null) {
                self.last = new_node;
            }
            self.len += 1;
        }

        pub fn appendLeft(self: *Self, data: T) error{OutOfMemory}!void {
            const new_node = try self.createNewNode(data);
            self.appendLeftNode(new_node);
        }

        pub fn popNode(self: *Self) Node {
            const last_node = self.last orelse @panic("Trying to popNode while linked list is empty");

            if (last_node.prev) |prev_node| {
                prev_node.next = null;
                self.last = prev_node;
                last_node.prev = null;
            }else{
                self.first = null;
                self.last = null;
            }
            self.len -= 1;

            return last_node;
        }

        pub fn pop(self: *Self) !T {
            const last_node = try self.pop_node();
            const data = last_node.data;
            self.allocator.destroy(last_node);
            return data;
        }

        pub fn popLeftNode(self: *Self) Node {
            const first_node = self.first orelse @panic("Trying to popLeftNode while linked list is empty");

            if (first_node.next) |next_node| {
                next_node.prev = null;
                self.first = next_node;
                first_node.next = null;
            }else{
                self.first = null;
                self.last = null;
            }
            self.len -= 1;
            return first_node;
        }

        pub fn popLeft(self: *Self) T {
            const first_node = self.popLeftNode();

            const data = first_node.data;
            self.allocator.destroy(first_node);
            return data;
        }

        pub fn is_empty(self: *Self) bool {
            return (self.len == 0);
        }

        pub fn clear(self: *Self) void {
            const allocator = self.allocator;
            var node = self.first;
            while (node) |n| {
                node = n.next;
                allocator.destroy(n);
            }
            self.* = Self.init(allocator);
        }

        pub fn extend(self: *Self, list: *Self) void {
            const f_node2 = list.first orelse return;
            const l_node2 = list.last.?;
            const new_len = list.len;

            list.first = null;
            list.last = null;
            list.len = 0;

            if (self.last) |l_node| {
                f_node2.prev = l_node;
                l_node.next = f_node2;
            }else{
                self.first = f_node2;
            }

            self.last = l_node2;
            self.len += new_len;
        }

        const Self = @This();
    };
}
