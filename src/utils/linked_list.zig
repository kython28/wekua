const std = @import("std");

pub fn LinkedList(comptime T: type) type {
    return struct {
        pub const Node = struct {
            prev: ?*Node = null,
            next: ?*Node = null,
            data: T,

            pub fn init(allocator: std.mem.Allocator, data: T) error{OutOfMemory}!*Node {
                const new_node = try allocator.create(Node);
                errdefer allocator.destroy(new_node);

                new_node.* = Node{
                    .data = data
                };

                return new_node;
            }

            pub fn deinit(self: *Node, allocator: std.mem.Allocator) void {
                if (self.prev != null or self.next != null) {
                    @panic("Node is not dispatched");
                }

                allocator.destroy(self);
            }

            pub fn dispatch(self: *Node) void {
                const next_node = self.next;
                const prev_node = self.prev;

                if (prev_node) |pn| {
                    pn.next = next_node;
                    self.prev = null;
                }

                if (next_node) |nn| {
                    nn.prev = prev_node;
                    self.next = null;
                }
            }
        };

        head: ?*Node = null,
        tail: ?*Node = null,
        size: usize = 0,

        pub fn append_node(self: *Self, node: *Node) void {
            const tail = self.tail;
            node.prev = tail;

            if (tail) |t| {
                t.next = node;
            }

            if (self.head == null) {
                self.head = node;
            }

            self.tail = node;
            self.size += 1;
        }

        pub fn append(self: *Self, allocator: std.mem.Allocator, data: T) error{OutOfMemory}!*T {
            const new_node = try Node.init(allocator, data);
            errdefer new_node.deinit();

            self.append_node(new_node);

            return &new_node.data;
        }

        pub fn appendleft_node(self: *Self, node: *Node) void {
            const head = self.head;
            node.next = head;

            if (head) |h| {
                h.prev = node;
            }

            if (self.tail == null) {
                self.tail = node;
            }

            self.head = node;
            self.size += 1;
        }

        pub fn appendleft(self: *Self, allocator: std.mem.Allocator, data: T) error{OutOfMemory}!*T {
            const new_node = try Node.init(allocator, data);
            errdefer new_node.deinit();

            self.appendleft_node(new_node);
            return &new_node.data;
        }

        pub fn pop_node(self: *Self) ?*Node {
            const tail = self.tail;
            if (tail) |t| {
                const prev_node = t.prev;
                self.tail = prev_node;

                if (prev_node) |pn| {
                    pn.next = null;
                }else{
                    self.head = null;
                }

                t.prev = null;
                self.size -= 1;
                return t;
            }

            return null;
        }

        pub fn pop(self: *Self) ?T {
            const node = self.pop_node() orelse return null;
            const value = node.data;
            node.deinit(self.allocator);
            return value;
        }

        pub fn popleft_node(self: *Self) ?*Node {
            const head = self.head;
            if (head) |h| {
                const next_node = h.next;
                self.head = next_node;

                if (next_node) |nn| {
                    nn.prev = null;
                }else{
                    self.tail = null;
                }

                h.next = null;
                self.size -= 1;
                return h;
            }

            return null;
        }

        pub fn popleft(self: *Self) ?T {
            const node = self.popleft_node() orelse return null;
            const value = node.data;
            node.deinit(self.allocator);
            return value;
        }

        pub fn destroy_node(self: *Self, allocator: std.mem.Allocator, node: *Node) void {
            const prev_node = node.prev;
            const next_node = node.next;

            if (prev_node) |pn| {
                pn.next = next_node;
            }

            if (next_node) |nn| {
                nn.prev = prev_node;
            }

            node.prev = null;
            node.next = null;
            node.deinit(allocator);

            if (self.head == node) {
                self.head = next_node;
            }

            if (self.tail == node) {
                self.tail = prev_node;
            }

            self.size -= 1;
        }

        pub fn reset(self: *Self, allocator: std.mem.Allocator) void {
            var node = self.head;
            while (node) |n| {
                const next_node = n.next;

                n.prev = null;
                n.next = null;
                n.deinit(allocator);
                node = next_node;
            }

            self.head = null;
            self.tail = null;
            self.size = 0;
        }

        pub fn dispatch(self: *Self, node: *Node) void {
            if (self.size == 0) {
                @panic("Dispatching from an empty list");
            }

            if (self.head == node) {
                self.head = node.next;
            }

            if (self.tail == node) {
                self.tail = node.prev;
            }

            node.dispatch();
            self.size -= 1;
        }

        const Self = @This();
    };
}

test "LinkedList - basic append and pop operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var list = LinkedList(i32).init(allocator);
    defer list.reset();

    // Test empty list
    try testing.expect(list.size == 0);
    try testing.expect(list.head == null);
    try testing.expect(list.tail == null);
    try testing.expect(list.pop() == null);

    // Test append
    _ = try list.append(1);
    try testing.expect(list.size == 1);
    try testing.expect(list.head != null);
    try testing.expect(list.tail != null);
    try testing.expect(list.head == list.tail);
    try testing.expect(list.head.?.data == 1);

    // Test append multiple
    _ = try list.append(2);
    _ = try list.append(3);
    try testing.expect(list.size == 3);
    try testing.expect(list.head.?.data == 1);
    try testing.expect(list.tail.?.data == 3);

    // Test pop
    const popped = list.pop();
    try testing.expect(popped == 3);
    try testing.expect(list.size == 2);
    try testing.expect(list.tail.?.data == 2);
}

test "LinkedList - appendleft and popleft operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var list = LinkedList(i32).init(allocator);
    defer list.reset();

    // Test appendleft
    _ = try list.appendleft(1);
    try testing.expect(list.size == 1);
    try testing.expect(list.head.?.data == 1);
    try testing.expect(list.tail.?.data == 1);

    // Test appendleft multiple
    _ = try list.appendleft(2);
    _ = try list.appendleft(3);
    try testing.expect(list.size == 3);
    try testing.expect(list.head.?.data == 3);
    try testing.expect(list.tail.?.data == 1);

    // Test popleft
    const popped = list.popleft();
    try testing.expect(popped == 3);
    try testing.expect(list.size == 2);
    try testing.expect(list.head.?.data == 2);
}

test "LinkedList - mixed operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var list = LinkedList(i32).init(allocator);
    defer list.reset();

    // Mix append and appendleft
    _ = try list.append(1);        // [1]
    _ = try list.appendleft(0);    // [0, 1]
    _ = try list.append(2);        // [0, 1, 2]
    _ = try list.appendleft(-1);   // [-1, 0, 1, 2]

    try testing.expect(list.size == 4);
    try testing.expect(list.head.?.data == -1);
    try testing.expect(list.tail.?.data == 2);

    // Mix pop and popleft
    try testing.expect(list.pop() == 2);      // [-1, 0, 1]
    try testing.expect(list.popleft() == -1);    // [0, 1]
    try testing.expect(list.size == 2);
    try testing.expect(list.head.?.data == 0);
    try testing.expect(list.tail.?.data == 1);
}

test "LinkedList - empty list to single element transitions" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var list = LinkedList(i32).init(allocator);
    defer list.reset();

    // Empty -> Single via append
    _ = try list.append(42);
    try testing.expect(list.size == 1);
    try testing.expect(list.head == list.tail);
    try testing.expect(list.head.?.data == 42);

    // Single -> Empty via pop
    try testing.expect(list.pop() == 42);
    try testing.expect(list.size == 0);
    try testing.expect(list.head == null);
    try testing.expect(list.tail == null);

    // Empty -> Single via appendleft
    _ = try list.appendleft(24);
    try testing.expect(list.size == 1);
    try testing.expect(list.head == list.tail);
    try testing.expect(list.head.?.data == 24);

    // Single -> Empty via popleft
    try testing.expect(list.popleft() == 24);
    try testing.expect(list.size == 0);
    try testing.expect(list.head == null);
    try testing.expect(list.tail == null);
}

test "LinkedList - node management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var list = LinkedList(i32).init(allocator);
    defer list.reset();

    // Test append_node and pop_node
    const node1 = try LinkedList(i32).Node.init(allocator, 100);
    list.append_node(node1);
    try testing.expect(list.size == 1);
    try testing.expect(list.head.?.data == 100);

    const popped_node = list.pop_node();
    try testing.expect(popped_node != null);
    try testing.expect(popped_node.?.data == 100);
    try testing.expect(list.size == 0);
    popped_node.?.deinit(allocator);

    // Test appendleft_node and popleft_node
    const node2 = try LinkedList(i32).Node.init(allocator, 200);
    list.appendleft_node(node2);
    try testing.expect(list.size == 1);
    try testing.expect(list.head.?.data == 200);

    const popleft_node = list.popleft_node();
    try testing.expect(popleft_node != null);
    try testing.expect(popleft_node.?.data == 200);
    try testing.expect(list.size == 0);
    popleft_node.?.deinit(allocator);
}

test "LinkedList - reset functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var list = LinkedList(i32).init(allocator);
    defer list.reset();

    // Add multiple elements
    _ = try list.append(1);
    _ = try list.append(2);
    _ = try list.append(3);
    _ = try list.appendleft(0);
    
    try testing.expect(list.size == 4);

    // Reset should clear everything
    list.reset();
    try testing.expect(list.size == 0);
    try testing.expect(list.head == null);
    try testing.expect(list.tail == null);

    // Should be able to use again after reset
    _ = try list.append(42);
    try testing.expect(list.size == 1);
    try testing.expect(list.head.?.data == 42);
}

test "LinkedList - with different types" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test with strings
    var string_list = LinkedList([]const u8).init(allocator);
    defer string_list.reset();

    _ = try string_list.append("hello");
    _ = try string_list.append("world");
    try testing.expect(string_list.size == 2);
    try testing.expect(std.mem.eql(u8, string_list.head.?.data, "hello"));
    try testing.expect(std.mem.eql(u8, string_list.tail.?.data, "world"));

    // Test with floats
    var float_list = LinkedList(f64).init(allocator);
    defer float_list.reset();

    _ = try float_list.append(3.14);
    _ = try float_list.appendleft(2.71);
    try testing.expect(float_list.size == 2);
    try testing.expect(float_list.head.?.data == 2.71);
    try testing.expect(float_list.tail.?.data == 3.14);
}

