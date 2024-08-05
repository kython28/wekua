const std = @import("std");

const _w_linked_list_node = struct {
    data: ?*anyopaque,

    prev: ?*_w_linked_list_node,
    next: ?*_w_linked_list_node
};

pub const Node = *_w_linked_list_node;

pub const errors = error {
    CannotHasElementsWhenRealasing,
    LinkedListEmpty
};

// pub const wLinkedList = struct {
allocator: std.mem.Allocator,
first: ?Node,
last: ?Node,
len: usize,

pub fn init(allocator: std.mem.Allocator) wLinkedList {
    return wLinkedList{
        .allocator = allocator,
        .first = null,
        .last = null,
        .len = 0
    };
}

pub fn append(self: *wLinkedList, data: ?*anyopaque) !void {
    const new_node = try self.allocator.create(_w_linked_list_node);

    if (self.last) |last_node| {
        last_node.next = new_node;
    }

    new_node.data = data;
    new_node.prev = self.last;
    new_node.next = null;

    self.last = new_node;

    if (self.first == null) {
        self.first = new_node;
    }

    self.len += 1;
}

pub fn appendleft(self: *wLinkedList, data: ?*anyopaque) !void {
    const new_node = try self.allocator.create(_w_linked_list_node);

    if (self.first) |first_node| {
        first_node.prev = new_node;
    }

    new_node.data = data;
    new_node.prev = null;
    new_node.next = self.first;

    self.first = new_node;
    if (self.last == null) {
        self.last = new_node;
    }
    self.len += 1;
}

pub fn pop(self: *wLinkedList) !?*anyopaque {
    const last_node = self.last orelse return errors.LinkedListEmpty;

    if (last_node.prev) |prev_node| {
        prev_node.next = null;
        self.last = prev_node;
    }else{
        self.first = null;
        self.last = null;
    }

    const data = last_node.data;
    self.allocator.destroy(last_node);
    self.len -= 1;
    return data;
}

pub fn popleft(self: *wLinkedList) !?*anyopaque {
    const first_node = self.first orelse return errors.LinkedListEmpty;

    if (first_node.next) |next_node| {
        next_node.prev = null;
        self.first = next_node;
    }else{
        self.first = null;
        self.last = null;
    }

    const data = first_node.data;
    self.allocator.destroy(first_node);
    self.len -= 1;
    return data;
}

pub fn is_empty(self: *wLinkedList) bool {
    return (self.len == 0);
}
// };

const wLinkedList = @This();
