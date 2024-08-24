const std = @import("std");
const cl = @import("opencl");

const wLinkedList = @import("../../utils/linked_list.zig");
const wTensor = @import("dtypes.zig").wTensor;
const command_queue_m = @import("../../core/command_queue.zig");
const wCommandQueue = command_queue_m.wCommandQueue;

const wQueue = @import("../../utils/queue.zig");

const clEventArray = std.ArrayList(cl.event.cl_event);
const UserCallbacksArray = std.ArrayList(*const event_callback);
const UserDataForCallbacksArray = std.ArrayList(?*anyopaque);
pub const event_callback = fn (allocator: std.mem.Allocator, user_data: ?*anyopaque) void;

pub const wTensorEventType = enum {
    write,
    read
};

const _w_tensor_event = struct {
    ctx_queue: *wQueue,
    allocator: std.mem.Allocator,
    command_queue: wCommandQueue,

    read_events: ?*clEventArray,
    write_event: ?cl.event.cl_event,

    mutex: *std.Thread.Mutex,
    condition: *std.Thread.Condition,
    event_type: wTensorEventType,
    events_finalized: usize,

    finalized: bool,

    callbacks: *UserCallbacksArray,
    user_datas: *UserDataForCallbacksArray
};

pub const wTensorEvent = *_w_tensor_event;

pub fn acquire_tensor(tensor: wTensor, event_type: wTensorEventType) ?[]const cl.event.cl_event {
    const mutex = &tensor.mutex; 
    mutex.lock();
    errdefer mutex.unlock();

    const tensor_events = tensor.events;
    var node = tensor_events.last;

    if (node == null) return null;

    var tensor_event: wTensorEvent = @alignCast(@ptrCast(node.?.data));
    if (event_type == .write) {
        return switch (tensor_event.event_type) {
            .read => tensor_event.read_events.?.items,
            .write => @as([*]cl.event.cl_event, @ptrCast(&tensor_event.write_event.?))[0..1]
        };
    }

    var event: ?[]const cl.event.cl_event = null;
    loop: while (node != null) {
        tensor_event = @alignCast(@ptrCast(node.?.data.?));
        switch (tensor_event.event_type) {
            .read => node = node.?.prev,
            .write => {
                event = @as([*]const cl.event.cl_event, @ptrCast(&tensor_event.write_event.?))[0..1];
                break :loop;
            }
        }
    }

    return event;
}

pub fn concatenate_events(
    allocator: std.mem.Allocator, events_array: []const ?[]const cl.event.cl_event
) !?[]cl.event.cl_event {
    var final_array_len: usize = 0;
    for (events_array) |events| {
        if (events) |v| {
            final_array_len += v.len;
        }
    }

    if (final_array_len == 0) return null;

    const final_array = try allocator.alloc(cl.event.cl_event, final_array_len);

    var offset: usize = 0;
    for (events_array) |events| {
        if (events) |v| {
            const arr_len = v.len;
            @memcpy(final_array[offset..(offset + arr_len)], v);
            offset += arr_len;
        }
    }

    return final_array;
}

fn tensor_event_callback(_: cl.event.cl_event, event_command_status: i32, user_data: ?*anyopaque) callconv(.C) void {
    const event_status: cl.event.enums.execution_status = @enumFromInt(event_command_status);
    if (event_status != .complete) unreachable;

    const node: wLinkedList.Node = @alignCast(@ptrCast(user_data.?));
    const events_node: wLinkedList.Node = @alignCast(@ptrCast(node.data.?));
    const tensor_event: wTensorEvent = @alignCast(@ptrCast(events_node.data.?));

    tensor_event.ctx_queue.put_node(node);
}

fn push_new_event_to_callback(event: cl.event.cl_event, tensor_event_node: wLinkedList.Node, ctx_queue: *wQueue) !void {
    const ctx_queue_ll = &ctx_queue.queue;
    const new_node = try ctx_queue_ll.create_new_node(tensor_event_node);
    errdefer ctx_queue_ll.release_node(new_node);

    try cl.event.set_callback(event, .complete, &tensor_event_callback, new_node);
}

fn create_new_tensor_event(
    command_queue: wCommandQueue, tensor: wTensor,
    callback: ?*const event_callback, user_data: ?*anyopaque,
    events: *wLinkedList, event: cl.event.cl_event,
    event_type: wTensorEventType
) !void {
    const allocator = command_queue.allocator;
    const tensor_event: wTensorEvent = try allocator.create(_w_tensor_event);
    errdefer allocator.destroy(tensor_event);

    tensor_event.allocator = allocator;
    tensor_event.command_queue = command_queue;
    tensor_event.events_finalized = 0;
    tensor_event.finalized = false;
    tensor_event.event_type = event_type;
    
    const callbacks = try allocator.create(UserCallbacksArray);
    errdefer allocator.destroy(callbacks);
    callbacks.* = UserCallbacksArray.init(allocator);

    const user_datas = try allocator.create(UserDataForCallbacksArray);
    errdefer allocator.destroy(user_datas);
    user_datas.* = UserDataForCallbacksArray.init(allocator);
    errdefer {
        callbacks.deinit();
        user_datas.deinit();
    }

    tensor_event.callbacks = callbacks;
    tensor_event.user_datas = user_datas;

    if (callback) |v| {
        try callbacks.append(v);
        errdefer {
            _ = callbacks.pop();
        }
        try user_datas.append(user_data);
    }
    errdefer {
        if (callback) |_| {
            _ = callbacks.pop();
            _ = user_datas.pop();
        }
    }

    tensor_event.mutex = &tensor.mutex;
    tensor_event.condition = &tensor.condition;

    const ctx_queue = &tensor.context.queue;
    tensor_event.ctx_queue = ctx_queue;

    try events.append(tensor_event);
    errdefer {
        _ = events.pop() catch unreachable;
    }

    switch (event_type) {
        .read => {
            const read_events_array: *clEventArray = try allocator.create(clEventArray);
            read_events_array.* = clEventArray.init(allocator);
            errdefer {
                read_events_array.deinit();
                allocator.destroy(read_events_array);
            }
            try read_events_array.append(event);
            
            tensor_event.read_events = read_events_array;
            tensor_event.write_event = null;
        },
        .write => {
            tensor_event.read_events = null;
            tensor_event.write_event = event;
        }
    }
    errdefer {
        if (event_type == .read) {
            const array = tensor_event.read_events.?;
            array.deinit();
            allocator.destroy(array);
        }
    }

    try push_new_event_to_callback(event, events.last.?, ctx_queue);
}


pub fn release_tensor_event(tensor_event: wTensorEvent) !void {
    const allocator = tensor_event.allocator;
    switch (tensor_event.event_type) {
        .read => {
            const read_events = tensor_event.read_events.?;
            for (read_events.items) |event| {
                try cl.event.release(event);
            }

            read_events.deinit();
            allocator.destroy(read_events);
        },
        .write => {
            try cl.event.release(tensor_event.write_event.?);
        },
    }

    tensor_event.callbacks.deinit();
    tensor_event.user_datas.deinit();
    allocator.destroy(tensor_event.callbacks);
    allocator.destroy(tensor_event.user_datas);
    allocator.destroy(tensor_event);
}

pub fn register_new_event(
    command_queue: wCommandQueue, tensor: wTensor,
    callback: ?*const event_callback, user_data: ?*anyopaque,
    event: cl.event.cl_event, event_type: wTensorEventType
) !void {
    const events = &tensor.events;

    if (event_type == .write) {
        try create_new_tensor_event(
            command_queue, tensor, callback, user_data, events, event, event_type
        );
        return;
    }

    if (events.last) |node| {
        const te: wTensorEvent = @alignCast(@ptrCast(node.data.?));

        switch (te.event_type) {
            .read => {
                const read_events = te.read_events.?;
                const total_read_events = read_events.items.len;
                if (te.events_finalized < total_read_events and total_read_events < 256) {
                    try read_events.append(event);
                    errdefer {
                        _ = read_events.pop();
                    }
                    if (callback) |v| {
                        try te.callbacks.append(v);
                        errdefer {
                            _ = te.callbacks.pop();
                        }
                        try te.user_datas.append(user_data);
                    }
                    errdefer {
                        _ = te.callbacks.pop();
                        _ = te.user_datas.pop();
                    }

                    try push_new_event_to_callback(event, node, &tensor.context.queue);
                    return;
                }
            },
            .write => {}
        }
    }
    try create_new_tensor_event(
        command_queue, tensor, callback, user_data, events, event, event_type
    );
}
