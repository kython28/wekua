const std = @import("std");

const _w_linked_list_node = struct {
    data: ?*anyopaque,

    prev: ?*_w_linked_list_node,
    next: ?*_w_linked_list_node
};

pub const wLinkedListNode = *_w_linked_list_node;

pub const errors = error {
    CannotHasElementsWhenRealasing,
    LinkedListEmpty
};

const _w_linked_list = struct {
    allocator: std.mem.Allocator,
    first: ?wLinkedListNode,
    last: ?wLinkedListNode
};

pub const wLinkedList = *_w_linked_list;

pub fn create(allocator: std.mem.Allocator) !wLinkedList {
    const linked_list = try allocator.create(_w_linked_list);
    linked_list.allocator = allocator;
    linked_list.first = null;
    linked_list.last = null;

    return linked_list;
}

pub fn append(linked_list: wLinkedList, data: ?*anyopaque) !void {
    const new_node = try linked_list.allocator.create(_w_linked_list_node);

    if (linked_list.last) |last_node| {
        last_node.next = new_node;
    }

    new_node.data = data;
    new_node.prev = linked_list.last;
    new_node.next = null;

    linked_list.last = new_node;

    if (linked_list.first == null) {
        linked_list.first = new_node;
    }
}

pub fn appendleft(linked_list: wLinkedList, data: ?*anyopaque) !void {
    const new_node = try linked_list.allocator.create(_w_linked_list_node);

    if (linked_list.first) |first_node| {
        first_node.prev = new_node;
    }

    new_node.data = data;
    new_node.prev = null;
    new_node.next = linked_list.first;

    linked_list.first = new_node;
    if (linked_list.last == null) {
        linked_list.last = new_node;
    }
}

pub fn pop(linked_list: wLinkedList) !?*anyopaque {
    const last_node = linked_list.last orelse return errors.LinkedListEmpty;

    if (last_node.prev) |prev_node| {
        prev_node.next = null;
        linked_list.last = prev_node;
    }else{
        linked_list.first = null;
        linked_list.last = null;
    }

    const data = last_node.data;
    linked_list.allocator.destroy(last_node);
    return data;
}

pub fn popleft(linked_list: wLinkedList) !?*anyopaque {
    const first_node = linked_list.first orelse return errors.LinkedListEmpty;
    
    if (first_node.next) |next_node| {
        next_node.prev = null;
        linked_list.first = next_node;
    }else{
        linked_list.first = null;
        linked_list.last = null;
    }

    const data = first_node.data;
    linked_list.allocator.destroy(first_node);
    return data;
}

pub fn is_empty(linked_list: wLinkedList) bool {
    return (linked_list.first == null);
}

pub fn release(linked_list: wLinkedList) !void {
    if (linked_list.first) |_| {
        return errors.CannotHasElementsWhenRealasing;
    }
    linked_list.allocator.destroy(linked_list);
}
