const wekua = @import("wekua");
const wLinkedList = wekua.utils.wLinkedList;

const std = @import("std");

const test_allocator = std.testing.allocator;

test "create and release" {
    const list: wLinkedList = wLinkedList.init(test_allocator);

    try std.testing.expect(list.first == null);
    try std.testing.expect(list.last == null);
    try std.testing.expect(list.len == 0);
}

test "append" {
    var list = wLinkedList.init(test_allocator);

    const elements = [_]usize{10, 20, 100, 10, 10, 20};
    for (elements) |elem| {
        try list.append(
            @ptrFromInt(elem)
        );
    }

    var node = list.first;
    var elements_total: usize = 0;
    while (node != null) {
        try std.testing.expect(@intFromPtr(node.?.data) == elements[elements_total]);
        elements_total += 1;

        const next_node = node.?.next;
        test_allocator.destroy(node.?);

        node = next_node;
    }
    list.first = null;
    list.last = null;

    try std.testing.expect(elements_total == elements.len);
}

test "appendleft" {
    var list = wLinkedList.init(test_allocator);

    const elements = [_]usize{10, 20, 100, 10, 10, 20};
    for (elements) |elem| {
        try list.appendleft(
            @ptrFromInt(elem)
        );
    }

    var node = list.last;
    var elements_total: usize = 0;
    while (node != null) {
        try std.testing.expect(@intFromPtr(node.?.data) == elements[elements_total]);
        elements_total += 1;

        const prev_node = node.?.prev;
        test_allocator.destroy(node.?);

        node = prev_node;
    }
    list.first = null;
    list.last = null;

    try std.testing.expect(elements_total == elements.len);
}

test "pop and popleft" {
    var list = wLinkedList.init(test_allocator);

    try std.testing.expectError(wLinkedList.errors.LinkedListEmpty, list.pop());
    try std.testing.expectError(wLinkedList.errors.LinkedListEmpty, list.popleft());

    const elements = [_]usize{2, 3, 10, 40};
    for (elements) |elem| {
        try list.append(
            @ptrFromInt(elem)
        );
    }

    var value: usize = @intFromPtr((try list.popleft()).?);
    try std.testing.expect(value == 2);

    value = @intFromPtr((try list.pop()).?);
    try std.testing.expect(value == 40);

    value = @intFromPtr((try list.popleft()).?);
    try std.testing.expect(value == 3);

    value = @intFromPtr((try list.pop()).?);
    try std.testing.expect(value == 10);

    try std.testing.expect(list.first == null);
    try std.testing.expect(list.last == null);
}

test "is_empty" {
    var list = wLinkedList.init(test_allocator);

    try std.testing.expect(list.is_empty());

    try list.append(@ptrFromInt(3));
    try std.testing.expect(!list.is_empty());

    _ = try list.pop();
    try std.testing.expect(list.is_empty());
}
