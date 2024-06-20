const std = @import("std");
const cl = @import("opencl");

const linked_list = @import("../../utils/linked_list.zig");
const wTensor = @import("dtypes.zig").wTensor;
const command_queue_m = @import("../../core/command_queue.zig");
const wCommandQueue = command_queue_m.wCommandQueue;

const wMutex = @import("../../utils/mutex.zig").wMutex;
const wCondition = @import("../../utils/condition.zig").wCondition;
const wQueue = @import("../../utils/queue.zig").wQueue;

const clEventArray = std.ArrayList(cl.event.cl_event);
pub const event_callback = fn (allocator: *const std.mem.Allocator, user_data: ?*anyopaque) void;

pub const wTensorEventType = enum {
    write,
    read
};

const _w_tensor_event = struct {
    ctx_queue: *wQueue,
    allocator: *const std.mem.Allocator,
    command_queue: wCommandQueue,
    read_events: ?*clEventArray,
    write_event: ?cl.event.cl_event,
    event: cl.event.cl_event,

    mutex: *wMutex,
    condition: *wCondition,
    event_type: wTensorEventType,
    events_finalized: usize,

    callback: ?*const event_callback,
    user_data: ?*anyopaque
};

pub const wTensorEvent = *_w_tensor_event;

pub fn acquire_tensor(tensor: wTensor, event_type: wTensorEventType) ?cl.event.cl_event {
    const mutex = tensor.mutex; 
    mutex.lock();
    errdefer mutex.unlock();

    const events = tensor.events;
    var node = events.last;
    var event: ?cl.event.cl_event = null;

    if (node == null) return null;

    var tensor_event: wTensorEvent = @alignCast(@ptrCast(node.?.data));
    if (event_type == .write) return tensor_event.event.?;

    loop: while (node != null) {
        tensor_event = @alignCast(@ptrCast(node.?.data.?));
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

    tensor_event.ctx_queue.put(node) catch unreachable;
}

fn create_new_tensor_event(
    command_queue: wCommandQueue, tensor: wTensor,
    callback: ?*const event_callback, user_data: ?*anyopaque,
    events: linked_list.wLinkedList, event: cl.event.cl_event,
    event_type: wTensorEventType
) !void {
    const allocator = command_queue.allocator;
    const tensor_event: wTensorEvent = try allocator.create(_w_tensor_event);
    errdefer allocator.destroy(tensor_event);

    tensor_event.allocator = allocator;
    tensor_event.command_queue = command_queue;
    tensor_event.events_finalized = 0;
    tensor_event.event_type = event_type;
    
    tensor_event.user_data = user_data;
    tensor_event.callback = callback;

    tensor_event.mutex = tensor.mutex;
    tensor_event.condition = tensor.condition;
    tensor_event.ctx_queue = tensor.context.queue;

    try linked_list.append(events, tensor_event);
    errdefer {
        _ = linked_list.pop(events) catch unreachable;
    }

    while (!command_queue_m.inc_event_counter(command_queue, 2, false)) {
        tensor.condition.wait(tensor.mutex);
    }
    errdefer command_queue_m.dec_event_counter(command_queue, 2) catch unreachable;

    const new_event: cl.event.cl_event = try cl.event.create_user_event(command_queue.ctx);
    errdefer cl.event.release(new_event) catch unreachable;
    tensor_event.event = new_event;

    switch (event_type) {
        .read => {
            const read_events_array: *clEventArray = try allocator.create(clEventArray);
            read_events_array.* = clEventArray.init(allocator.*);
            errdefer {
                read_events_array.deinit();
                allocator.destroy(read_events_array);
            }
            try read_events_array.append(event);
            try cl.event.set_callback(event, .complete, &tensor_event_callback, events.last);
            
            tensor_event.read_events = read_events_array;
            tensor_event.write_event = null;
        },
        .write => {
            tensor_event.read_events = null;
            tensor_event.write_event = event;
            try cl.event.set_callback(event, .complete, &tensor_event_callback, events.last);
        }
    }
}

pub fn release_tensor_event(tensor_event: wTensorEvent) void {
    const command_queue = tensor_event.command_queue;
    const allocator = tensor_event.allocator;
    switch (tensor_event.event_type) {
        .read => {
            const read_events = tensor_event.read_events.?;
            for (read_events.items) |event| {
                cl.event.release(event) catch unreachable;
                command_queue_m.dec_event_counter(command_queue, 1) catch unreachable;
            }

            read_events.deinit();
            allocator.destroy(read_events);
        },
        .write => {
            cl.event.release(tensor_event.write_event.?) catch unreachable;
            command_queue_m.dec_event_counter(command_queue, 1) catch unreachable;
        },
    }

    cl.event.release(tensor_event.event) catch unreachable;
    command_queue_m.dec_event_counter(command_queue, 1) catch unreachable;

    allocator.destroy(tensor_event);
}

pub fn register_new_event(
    command_queue: wCommandQueue, tensor: wTensor,
    callback: ?*const event_callback, user_data: ?*anyopaque,
    event: cl.event.cl_event, event_type: wTensorEventType
) !void {
    const events = tensor.events;

    if (event_type == .write) {
        try create_new_tensor_event(command_queue, tensor, callback, user_data, events, event, event_type);
        return;
    }

    if (events.last) |node| {
        const te: wTensorEvent = @alignCast(@ptrCast(node.data.?));

        switch (te.event_type) {
            .read => {
                const read_events = te.read_events.?;
                const total_read_events = read_events.items.len;
                if (te.events_finalized < total_read_events and total_read_events < (command_queue.max_number_of_events/2)) {
                    try read_events.append(event);
                    while (!command_queue_m.inc_event_counter(command_queue, 1, false)) {
                        tensor.condition.wait(tensor.mutex);
                    }
                    return;
                }
            },
            .write => {}
        }
    }
    try create_new_tensor_event(command_queue, tensor, callback, user_data, events, event, event_type);
}
