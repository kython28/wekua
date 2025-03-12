const std = @import("std");
const cl = @import("opencl");
const wCommandQueue = @import("command_queue.zig");

const wQueue = @import("../utils/queue.zig");
const w_event = @import("event.zig");

allocator: std.mem.Allocator,
ctx: cl.context.cl_context,
command_queues: []wCommandQueue,

events_worker: std.Thread,
queue: wQueue,

releasing_events_worker: std.Thread,
releasing_events_queue: wQueue,

pub fn init(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    devices: []cl.device.cl_device_id
) !*wContext {
    const cl_ctx = try cl.context.create(properties, devices, null, null);
    errdefer cl.context.release(cl_ctx) catch unreachable;

    const context = try init_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn init_from_device_type(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    device_type: cl.device.enums.device_type
) !*wContext {
    const cl_ctx = try cl.context.create_from_type(properties, device_type, null, null);
    errdefer cl.context.release(cl_ctx) catch unreachable;

    const context = try init_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn create_from_best_device(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    device_type: cl.device.enums.device_type
) !*wContext {
    const platforms = try cl.platform.get_all(allocator);
    defer cl.platform.release_list(allocator, platforms);

    var best_device: ?cl.device.cl_device_id = null;
    var best_score: u64 = 0;
    for (platforms) |plat| {
        var num_devices: u32 = undefined;
        try cl.device.get_ids(plat.id.?, device_type, null, &num_devices);
        if (num_devices == 0) continue;
        
        const devices = try allocator.alloc(cl.device.cl_device_id, num_devices);
        defer {
            for (devices) |dev| {
                if (dev != best_device) {
                    cl.device.release(dev) catch unreachable;
                }
            }
            allocator.free(devices);
        }

        try cl.device.get_ids(plat.id.?, device_type, devices, null);
        var device_choosen_in_this_iteration: bool = false;
        for (devices) |device| {
            var clock_freq: u32 = undefined;
            try cl.device.get_info(
                device, cl.device.enums.device_info.max_clock_frequency, @sizeOf(u32), &clock_freq, null
            );

            var max_work_group_size: u64 = undefined;
            try cl.device.get_info(
                device, cl.device.enums.device_info.max_work_group_size, @sizeOf(u64), &max_work_group_size, null
            );

            var compute_units: u32 = undefined;
            try cl.device.get_info(
                device, cl.device.enums.device_info.partition_max_sub_devices, @sizeOf(u32),
                &compute_units, null
            );

            const score: u64 = clock_freq * max_work_group_size * compute_units;
            if (score > best_score or best_device == null) {
                if (best_device) |dev| {
                    if (!device_choosen_in_this_iteration) try cl.device.release(dev);
                    device_choosen_in_this_iteration = true;
                }
                best_device = device;
                best_score = score;
            }
        }
    }

    if (best_device == null) return error.DeviceNotFound;

    const cl_ctx = try cl.context.create(properties, &.{best_device.?}, null, null);
    errdefer cl.context.release(cl_ctx) catch unreachable;

    const context = try init_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn init_from_cl_context(
    allocator: std.mem.Allocator,
    cl_ctx: cl.context.cl_context
) !*wContext {
    var context = try allocator.create(wContext);
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
    context.command_queues = try wCommandQueue.init_multiples(allocator, context, devices);
    context.queue = wQueue.init(allocator);
    context.releasing_events_queue = wQueue.init(allocator);

    context.events_worker = try std.Thread.spawn(
        .{}, w_event.context_events_worker, .{allocator, &context.queue, &context.releasing_events_queue}
    );
    errdefer {
        context.queue.release();
        context.events_worker.join();
    }

    context.releasing_events_worker = try std.Thread.spawn(
        .{}, w_event.context_releasing_events_worker, .{allocator, &context.releasing_events_queue}
    );
    return context;
}

pub fn release(context: *wContext) void {
    const allocator = context.allocator;

    context.queue.release();
    context.events_worker.join();

    context.releasing_events_queue.release();
    context.releasing_events_worker.join();

    cl.context.release(context.ctx) catch unreachable;
    wCommandQueue.deinit_multiples(allocator, context.command_queues);

    allocator.destroy(context);
}

const wContext = @This();
