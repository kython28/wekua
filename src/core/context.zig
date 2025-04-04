const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("command_queue.zig");
const events_releaser = @import("events_releaser.zig");

pub const SupportedTypes: [10]type = .{ i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 };

pub fn getTypeId(comptime T: type) comptime_int {
    inline for (SupportedTypes, 0..) |t, i| {
        if (T == t) {
            return i;
        }
    }

    @compileError("Type not supported");
}

allocator: std.mem.Allocator,
ctx: cl.context.cl_context,
command_queues: []CommandQueue,
events_batch_queue: events_releaser.EventsBatchQueue,
events_releaser_thread: std.Thread,

pub fn init(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    devices: []cl.device.cl_device_id,
) !*Context {
    const cl_ctx = try cl.context.create(properties, devices, null, null);
    errdefer cl.context.release(cl_ctx);

    const context = try init_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn init_from_device_type(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    device_type: cl.device.enums.device_type,
) !*Context {
    const cl_ctx = try cl.context.create_from_type(properties, device_type, null, null);
    errdefer cl.context.release(cl_ctx);

    const context = try init_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn create_from_best_device(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.cl_context_properties,
    device_type: cl.device.enums.device_type,
) !*Context {
    const platforms = try cl.platform.get_all(allocator);
    defer cl.platform.release_list(allocator, platforms);

    var best_device: ?cl.device.cl_device_id = null;
    var best_score: u64 = 0;
    for (platforms) |plat| {
        var num_devices: u32 = undefined;
        cl.device.get_ids(plat.id.?, device_type, null, &num_devices) catch continue;
        if (num_devices == 0) continue;

        const devices = try allocator.alloc(cl.device.cl_device_id, num_devices);
        defer {
            for (devices) |dev| {
                if (dev != best_device) {
                    cl.device.release(dev);
                }
            }
            allocator.free(devices);
        }

        try cl.device.get_ids(plat.id.?, device_type, devices, null);
        var device_choosen_in_this_iteration: bool = false;
        for (devices) |device| {
            var max_work_group_size: u64 = undefined;
            try cl.device.get_info(
                device,
                cl.device.enums.device_info.max_work_group_size,
                @sizeOf(u64),
                &max_work_group_size,
                null,
            );

            var compute_units: u32 = undefined;
            try cl.device.get_info(
                device,
                cl.device.enums.device_info.partition_max_sub_devices,
                @sizeOf(u32),
                &compute_units,
                null,
            );

            const score: u64 = max_work_group_size * compute_units;
            if (score > best_score or best_device == null) {
                if (best_device) |dev| {
                    if (!device_choosen_in_this_iteration) cl.device.release(dev);
                    device_choosen_in_this_iteration = true;
                }
                best_device = device;
                best_score = score;
            }
        }
    }

    if (best_device == null) return error.DeviceNotFound;

    const cl_ctx = try cl.context.create(properties, &.{best_device.?}, null, null);
    errdefer cl.context.release(cl_ctx);

    const context = try init_from_cl_context(allocator, cl_ctx);
    return context;
}

pub fn init_from_cl_context(allocator: std.mem.Allocator, cl_ctx: cl.context.cl_context) !*Context {
    var context = try allocator.create(Context);
    errdefer allocator.destroy(context);

    var number_of_devices: u32 = undefined;
    try cl.context.get_info(cl_ctx, cl.context.enums.context_info.num_devices, @sizeOf(u32), &number_of_devices, null);

    const devices: []cl.device.cl_device_id = try allocator.alloc(cl.device.cl_device_id, @intCast(number_of_devices));
    defer allocator.free(devices);

    try cl.context.get_info(
        cl_ctx,
        cl.context.enums.context_info.devices,
        @sizeOf(cl.device.cl_device_id) * number_of_devices,
        @ptrCast(devices.ptr),
        null,
    );

    context.allocator = allocator;
    context.ctx = cl_ctx;
    context.command_queues = try CommandQueue.init_multiples(allocator, context, devices);
    errdefer CommandQueue.deinit_multiples(allocator, context.command_queues);

    context.events_batch_queue = events_releaser.EventsBatchQueue.init(allocator);
    {
        errdefer context.events_batch_queue.release();

        context.events_releaser_thread = try std.Thread.spawn(
            .{},
            events_releaser.eventsBatchReleaserWorker,
            .{&context.events_batch_queue},
        );
    }
    errdefer {
        context.events_batch_queue.release();
        context.events_releaser_thread.join();
    }

    return context;
}

pub fn release(context: *Context) void {
    const allocator = context.allocator;

    CommandQueue.deinit_multiples(allocator, context.command_queues);

    cl.context.release(context.ctx);

    context.events_batch_queue.release();
    context.events_releaser_thread.join();

    allocator.destroy(context);
}

const Context = @This();
