const std = @import("std");

const wLinkedList = @import("../utils/linked_list.zig");
const w_tensor_event = @import("../tensor/utils/event.zig");
const wTensorEvent = w_tensor_event.wTensorEvent;

const wQueue = @import("../utils/queue.zig");

fn release_previous_event(node: wLinkedList.Node, tensor_event: wTensorEvent) !bool {
    const allocator = tensor_event.allocator;
    if (node.prev) |prev_node| {
        const prev_tensor_event: wTensorEvent = @alignCast(@ptrCast(prev_node.data.?));
        if (!prev_tensor_event.finalized) {
            return false;
        }

        try w_tensor_event.release_tensor_event(prev_tensor_event);
        allocator.destroy(prev_node);

        node.prev = null;
    }

    if (tensor_event.callbacks.items.len > 0) {
        for (tensor_event.callbacks.items, tensor_event.user_datas.items) |c, d| {
            c(allocator, d);
        }
    }
    tensor_event.finalized = true;
    tensor_event.condition.broadcast();
    return true;
}

fn deal_with_new_tensor_event(
    allocator: std.mem.Allocator, ctx_queue_node: wLinkedList.Node, data: *anyopaque, pending_events: *wQueue
) !bool {
    const node: wLinkedList.Node = @alignCast(@ptrCast(data));
    const tensor_event: wTensorEvent = @alignCast(@ptrCast(node.data.?));
    const mutex = tensor_event.mutex;

    // if (!mutex.tryLock()) {
    //     return true;
    // }
    mutex.lock();
    defer mutex.unlock();

    tensor_event.events_finalized += 1;
    const finished: bool = switch (tensor_event.event_type) {
        .read => blk: {
            const cond: bool = (tensor_event.events_finalized == tensor_event.read_events.?.items.len);
            break :blk cond;
        },
        .write => true
    };

    if (finished) {
        tensor_event.condition.broadcast();
        pending_events.put_node(ctx_queue_node);
    }else{
        allocator.destroy(ctx_queue_node);
    }
    return false;
}

fn deal_with_old_tensor_events(pending_events: *wLinkedList) !void {
    var node: ?wLinkedList.Node = pending_events.first;
    var index: usize = 0;
    var length: usize = pending_events.len;
    while (index < length) {
        const tensor_event_queue_node: wLinkedList.Node = @alignCast(@ptrCast(node.?.data.?));
        const tensor_event: wTensorEvent = @alignCast(@ptrCast(tensor_event_queue_node.data.?));
        const mutex = tensor_event.mutex;

        while (!mutex.tryLock()) {
            std.time.sleep(1000000);
        }
        defer mutex.unlock();

        const next_node = node.?.next;
        const released = try release_previous_event(tensor_event_queue_node, tensor_event);
        if (released) {
            const prev_node = node.?.prev;
            if (prev_node) |pn| {
                pn.next = next_node;
            }else{
                pending_events.first = next_node;
            }

            if (next_node) |nn| {
                nn.prev = prev_node;
            }else{
                pending_events.last = prev_node;
            }

            pending_events.allocator.destroy(node.?);
            pending_events.len -= 1;
            length -= 1;
            index = 0;

            node = next_node orelse pending_events.first;
            continue;
        }else{
            node = next_node orelse pending_events.first;
        }
        index += 1;
    }
}

pub fn context_events_worker(allocator: std.mem.Allocator, queue: *wQueue, pending_events: *wQueue) void {
    while (true) {
        const last_node: ?wLinkedList.Node = queue.get_last_node(true) catch unreachable;
        if (last_node == null) break;

        const cl_user_data: ?*anyopaque = last_node.?.data;
        if (cl_user_data) |v| {
            const not_executed = deal_with_new_tensor_event(allocator, last_node.?, v, pending_events) catch unreachable;
            if (not_executed) queue.put_node(last_node.?);
        }else{
            allocator.destroy(last_node.?);
            break;
        }
    }
}

pub fn context_releasing_events_worker(allocator: std.mem.Allocator, queue: *wQueue) void {
    var pending_events: wLinkedList = wLinkedList.init(allocator);
    while (true) {
        const last_node: ?wLinkedList.Node = queue.get_last_node(true) catch unreachable;
        if (last_node) |v| {
            pending_events.append_node(v);
        }else{
            break;
        }

        deal_with_old_tensor_events(&pending_events) catch unreachable;
    }
}
