const std = @import("std");
const cl = @import("opencl");

const linked_list = @import("../utils/linked_list.zig");
const wTensor = @import("empty.zig").wTensor;
const command_queue_m = @import("../core/command_queue.zig");
const wCommandQueue = command_queue_m.wCommandQueue;

const clEventArray = std.ArrayList(?cl.event.cl_event);

pub const wTensorEventType = enum {
    write,
    read
};

const _w_tensor_event = struct {
    allocator: *const std.mem.Allocator,
    command_queue: wCommandQueue,
    read_events: ?*clEventArray,
    event: cl.event.cl_event,

    mutex: *std.Thread.Mutex,
    event_type: wTensorEventType,
    events_finalized: usize,
    can_release_event: bool
};

pub const wTensorEvent = *_w_tensor_event;

pub fn acquire_tensor(tensor: wTensor) ?cl.event.cl_event {
    const mutex = tensor.mutex; 
    mutex.lock();
    errdefer mutex.unlock();

    const events = tensor.events;
    var node = events.last;
    var event: ?cl.event.cl_event = null;
    loop: while (node != null) {
        const tensor_event: wTensorEvent = @alignCast(@ptrCast(node.?.data));
        switch (tensor_event.event_type) {
            .read => node = node.?.prev,
            .write => {
                    event = tensor_event.event;
                    break :loop;
            }
        }
    }

    return event;
}

fn tensor_event_callback(_: cl.event.cl_event, event_command_status: i32, user_data: ?*anyopaque) callconv(.C) void {
    const event_status: cl.event.enums.execution_status = @enumFromInt(event_command_status);
    if (event_status != .complete) unreachable;

    const node: linked_list.wLinkedListNode = @alignCast(@ptrCast(user_data.?));
    const tensor_event: wTensorEvent = @alignCast(@ptrCast(node.data.?));
    const mutex = tensor_event.mutex;

    mutex.lock();
    defer mutex.unlock();

    tensor_event.events_finalized += 1;
    const finished: bool = switch (tensor_event.event_type) {
        .read => (tensor_event.events_finalized == tensor_event.read_events.?.items.len),
        .write => true
    };

    if (finished) {
        if (node.prev) |prev_node| {
            release_tensor_event(@alignCast(@ptrCast(prev_node.data.?)));
            tensor_event.allocator.destroy(prev_node);

            node.prev = null;
        }
    }
}

fn create_new_tensor_event(
    command_queue: wCommandQueue, allocator: *const std.mem.Allocator,
    events: linked_list.wLinkedList, event: cl.event.cl_event, event_type: wTensorEventType,
    can_release_event: bool
) !void {
    const tensor_event: wTensorEvent = try allocator.create(_w_tensor_event);
    errdefer allocator.destroy(tensor_event);

    tensor_event.allocator = allocator;
    tensor_event.command_queue = command_queue;
    tensor_event.events_finalized = 0;
    tensor_event.event_type = event_type;

    const mutex: *std.Thread.Mutex = try allocator.create(std.Thread.Mutex);
    mutex.* = std.Thread.Mutex{};
    errdefer allocator.destroy(mutex);
    tensor_event.mutex = mutex;

    try linked_list.append(events, tensor_event);
    errdefer {
        _ = linked_list.pop(events) catch unreachable;
    }

    switch (event_type) {
        .read => {
            const read_events_array: *clEventArray = try allocator.create(clEventArray);
            read_events_array.* = clEventArray.init(allocator.*);
            errdefer {
                read_events_array.deinit();
                allocator.destroy(read_events_array);
            }
            if (can_release_event) {
                try read_events_array.append(event);
            }else{
                try read_events_array.append(null);
            }

            const new_event: cl.event.cl_event = try cl.event.create_user_event(command_queue.ctx);
            errdefer cl.event.release(new_event) catch unreachable;
            try cl.event.set_callback(event, cl.event.enums.execution_status.complete, &tensor_event_callback, events.last);
            
            tensor_event.read_events = read_events_array;
            tensor_event.event = new_event;
            tensor_event.can_release_event = true;
        },
        .write => {
            tensor_event.read_events = null;
            tensor_event.event = event;
            tensor_event.can_release_event = can_release_event;
        }
    }

    if (can_release_event) {
        command_queue_m.inc_event_counter(command_queue);
    }
}

pub fn release_tensor_event(tensor_event: wTensorEvent) void {
    const command_queue = tensor_event.command_queue;
    switch (tensor_event.event_type) {
        .read => {
            const read_events = tensor_event.read_events.?;
            for (read_events.items) |event| {
                if (event) |e| {
                    cl.event.release(e) catch unreachable;
                    command_queue_m.dec_event_counter(command_queue) catch unreachable;
                }
            }

            read_events.deinit();
        },
        .write => {},
    }

    if (tensor_event.can_release_event) {
        cl.event.release(tensor_event.event) catch unreachable;
        command_queue_m.dec_event_counter(command_queue) catch unreachable;
    }

    tensor_event.allocator.destroy(tensor_event.mutex);
    tensor_event.allocator.destroy(tensor_event);
}

pub fn register_new_event(
    command_queue: wCommandQueue, tensor: wTensor,
    event: cl.event.cl_event, event_type: wTensorEventType,
    can_release_event: bool
) !void {
    const allocator = command_queue.allocator;
    const events = tensor.events;

    if (event_type == .write) {
        try create_new_tensor_event(command_queue, allocator, events, event, event_type, can_release_event);
        return;
    }

    if (events.last) |node| {
        const te: wTensorEvent = @alignCast(@ptrCast(node.data.?));

        const mutex = te.mutex;
        mutex.lock();
        defer mutex.unlock();

        switch (te.event_type) {
            .read => {
                const read_events = te.read_events.?;
                const total_read_events = read_events.items.len;
                if (te.events_finalized <= total_read_events and total_read_events <= (command_queue.max_number_of_events/2)) {
                    if (can_release_event) {
                        try read_events.append(event);
                        command_queue_m.inc_event_counter(command_queue);
                    }else{
                        try read_events.append(null);
                    }
                }
            },
            .write => {}
        }
    }
    try create_new_tensor_event(command_queue, allocator, events, event, event_type, can_release_event);
}
