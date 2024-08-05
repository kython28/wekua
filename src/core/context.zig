const std = @import("std");
const cl = @import("opencl");
const command_queue = @import("command_queue.zig");

const wLinkedList = @import("../utils/linked_list.zig");
const w_tensor_event = @import("../tensor/utils/event.zig");
const wTensorEvent = w_tensor_event.wTensorEvent;

const wQueue = @import("../utils/queue.zig");

const _wcontext = struct {
    allocator: std.mem.Allocator,
    ctx: cl.context.cl_context,
    command_queues: []command_queue.wCommandQueue,

    worker: std.Thread,
    queue: wQueue
};

pub const wContext = *_wcontext;

fn release_previous_event(node: wLinkedList.Node, tensor_event: wTensorEvent) !bool {
    const allocator = tensor_event.allocator;
    const t_event_condition = tensor_event.condition;
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
    t_event_condition.broadcast();
    return true;
}

fn deal_with_new_tensor_event(data: *anyopaque, pending_events: *wLinkedList) !void {
    const node: wLinkedList.Node = @alignCast(@ptrCast(data));
    const tensor_event: wTensorEvent = @alignCast(@ptrCast(node.data.?));
    const mutex = tensor_event.mutex;

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
        const released = try release_previous_event(node, tensor_event);
        if (!released) {
            try pending_events.append(node);
        }
    }
}

fn deal_with_old_tensor_events(pending_events: *wLinkedList) !void {
    var node: ?wLinkedList.Node = pending_events.first;
    var stop_node: ?wLinkedList.Node = null;
    while (node != null) {
        const tensor_event_queue_node: wLinkedList.Node = @alignCast(@ptrCast(node.?.data.?));
        const tensor_event: wTensorEvent = @alignCast(@ptrCast(tensor_event_queue_node.data.?));
        const mutex = tensor_event.mutex;

        mutex.lock();
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

            stop_node = next_node;
            node = next_node orelse pending_events.first;
        }else{
            if (next_node == stop_node) break;
            node = next_node orelse pending_events.first;
        }
    }
}

fn context_events_worker(allocator: std.mem.Allocator, queue: *wQueue) void {
    var pending_events: wLinkedList = wLinkedList.init(allocator);

    while (true) {
        const cl_user_data: ?*anyopaque = queue.get(true) catch unreachable;
        if (cl_user_data) |v| {
            deal_with_new_tensor_event(v, &pending_events) catch unreachable;
        }else{
            break;
        }

        deal_with_old_tensor_events(&pending_events) catch unreachable;
    }
}

pub fn create(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    devices: []cl.device.cl_device_id
) !wContext {
    const cl_ctx = try cl.context.create(properties, devices, null, null);
    errdefer cl.context.release(cl_ctx) catch unreachable;

    const context = try create_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn create_from_device_type(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    device_type: cl.device.enums.device_type
) !wContext {
    const cl_ctx = try cl.context.create_from_type(properties, device_type, null, null);
    errdefer cl.context.release(cl_ctx) catch unreachable;

    const context = try create_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn create_from_cl_context(
    allocator: std.mem.Allocator,
    cl_ctx: cl.context.cl_context
) !wContext {
    const context: wContext = try allocator.create(_wcontext);
    errdefer allocator.destroy(context);

    var number_of_devices: u32 = undefined;
    try cl.context.get_info(cl_ctx, cl.context.enums.context_info.num_devices, @sizeOf(u32), &number_of_devices, null);

    const devices: []cl.device.cl_device_id = try allocator.alloc(cl.device.cl_device_id, @intCast(number_of_devices));
    defer allocator.free(devices);

    try cl.context.get_info(
        cl_ctx, cl.context.enums.context_info.devices, @sizeOf(cl.device.cl_device_id) * number_of_devices,
        @ptrCast(devices.ptr), null
    );


    context.allocator = allocator;
    context.ctx = cl_ctx;
    context.command_queues = try command_queue.create_multiple(allocator, cl_ctx, devices);
    context.queue = wQueue.init(allocator);

    context.worker = try std.Thread.spawn(.{}, context_events_worker, .{allocator, &context.queue});
    return context;
}

pub fn release(context: wContext) void {
    const allocator = context.allocator;

    context.queue.release();
    context.worker.join();

    cl.context.release(context.ctx) catch unreachable;
    command_queue.release_multiple(allocator, context.command_queues);

    allocator.destroy(context);
}

