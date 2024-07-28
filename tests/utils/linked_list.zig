const wekua = @import("wekua");
const linked_list = wekua.utils.linked_list;

const std = @import("std");

const test_allocator = std.testing.allocator;

test "create and release" {
    const list: linked_list.wLinkedList = try linked_list.create(test_allocator);

    try std.testing.expect(list.first == null);
    try std.testing.expect(list.last == null);

    try linked_list.release(list);
}

test "append" {
    const list = try linked_list.create(test_allocator);

    const elements = [_]usize{10, 20, 100, 10, 10, 20};
    for (elements) |elem| {
        try linked_list.append(
            list, @ptrFromInt(elem)
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
    try linked_list.release(list);
}

test "appendleft" {
    const list = try linked_list.create(test_allocator);

    const elements = [_]usize{10, 20, 100, 10, 10, 20};
    for (elements) |elem| {
        try linked_list.appendleft(
            list, @ptrFromInt(elem)
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
    try linked_list.release(list);
}

test "pop and popleft" {
    const list = try linked_list.create(test_allocator);

    try std.testing.expectError(linked_list.errors.LinkedListEmpty, linked_list.pop(list));
    try std.testing.expectError(linked_list.errors.LinkedListEmpty, linked_list.popleft(list));

    const elements = [_]usize{2, 3, 10, 40};
    for (elements) |elem| {
        try linked_list.append(
            list, @ptrFromInt(elem)
        );
    }

    var value: usize = @intFromPtr((try linked_list.popleft(list)).?);
    try std.testing.expect(value == 2);

    value = @intFromPtr((try linked_list.pop(list)).?);
    try std.testing.expect(value == 40);

    value = @intFromPtr((try linked_list.popleft(list)).?);
    try std.testing.expect(value == 3);

    value = @intFromPtr((try linked_list.pop(list)).?);
    try std.testing.expect(value == 10);

    try std.testing.expect(list.first == null);
    try std.testing.expect(list.last == null);
    try linked_list.release(list);
}

test "is_empty" {
    const list = try linked_list.create(test_allocator);

    try std.testing.expect(linked_list.is_empty(list));

    try linked_list.append(list, @ptrFromInt(3));
    try std.testing.expect(!linked_list.is_empty(list));

    _ = try linked_list.pop(list);
    try std.testing.expect(linked_list.is_empty(list));


    try linked_list.release(list);
}
