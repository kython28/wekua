const std = @import("std");
const cl = @import("opencl");
const command_queue = @import("command_queue.zig");

const linked_list = @import("../utils/linked_list.zig");
const w_tensor_event = @import("../tensor/utils/event.zig");

const wQueue = @import("../utils/queue.zig").wQueue;

const _wcontext = struct {
    allocator: *const std.mem.Allocator,
    ctx: cl.context.cl_context,
    command_queues: []command_queue.wCommandQueue,

    workers: [4]?std.Thread,
    queue: *wQueue
};

pub const wContext = *_wcontext;

fn context_events_worker(queue: *wQueue) void {
    while (true) {
        const cl_user_data: ?*anyopaque = queue.get(true) catch unreachable;
        if (cl_user_data == null) {
            break;
        }

        const node: linked_list.wLinkedListNode = @alignCast(@ptrCast(cl_user_data.?));
        const tensor_event: w_tensor_event.wTensorEvent = @alignCast(@ptrCast(node.data.?));
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
            cl.event.set_user_event_status(tensor_event.event, .complete) catch unreachable;
            const allocator = tensor_event.allocator;
            if (node.prev) |prev_node| {
                w_tensor_event.release_tensor_event(@alignCast(@ptrCast(prev_node.data.?)));
                allocator.destroy(prev_node);

                node.prev = null;
            }

            if (tensor_event.callback) |callback| {
                callback(allocator, tensor_event.user_data);
            }
            tensor_event.condition.broadcast();
        }
    }

}

pub fn create(
    allocator: *const std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    devices: []cl.device.cl_device_id
) !wContext {
    const cl_ctx = try cl.context.create(properties, devices, null, null);
    errdefer cl.context.release(cl_ctx) catch unreachable;

    const context = try create_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn create_from_device_type(
    allocator: *const std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    device_type: cl.device.enums.device_type
) !wContext {
    const cl_ctx = try cl.context.create_from_type(properties, device_type, null, null);
    errdefer cl.context.release(cl_ctx) catch unreachable;

    const context = try create_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn create_from_cl_context(
    allocator: *const std.mem.Allocator,
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
    context.queue = try allocator.create(wQueue);
    context.queue.* = try wQueue.init(allocator);

    errdefer {
        context.queue.release();
        allocator.destroy(context.queue);
    }

    var number_of_workers_alive: u8 = 0;
    errdefer {
        for (&context.workers) |_| {
            context.queue.put(null) catch unreachable;
        }

        for (&context.workers) |worker| {
            if (worker) |w| w.join();
        }
    }

    for (&context.workers) |*w| {
        w.* = try std.Thread.spawn(.{}, context_events_worker, .{context.queue});
        number_of_workers_alive += 1;
    }

    return context;
}

pub fn release(context: wContext) void {
    const allocator = context.allocator;

    for (context.workers) |_| {
        context.queue.put(null) catch unreachable;
    }

    for (context.workers) |w| {
        w.?.join();
    }
    context.queue.release();

    cl.context.release(context.ctx) catch unreachable;
    command_queue.release_multiple(allocator, context.command_queues);

    allocator.destroy(context.queue);
    allocator.destroy(context);
}

