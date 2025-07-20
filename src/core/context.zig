const std = @import("std");
const cl = @import("opencl");

const CommandQueue = @import("command_queue.zig");
// const events_releaser = @import("events_releaser.zig");

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
ctx: cl.context.Context,
command_queues: []CommandQueue,
// events_batch_queue: events_releaser.EventsBatchQueue,
// events_releaser_thread: std.Thread,

pub fn init(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.Properties,
    devices: []cl.device.DeviceId,
) !*Context {
    const cl_ctx = try cl.context.create(properties, devices, null, null);
    errdefer cl.context.release(cl_ctx);

    const context = try initFromClContext(allocator, cl_ctx);
    return context;
}

pub fn initFromDeviceType(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.Properties,
    device_type: cl.device.Type,
) !*Context {
    const cl_ctx = try cl.context.createFromType(properties, device_type, null, null);
    errdefer cl.context.release(cl_ctx);

    const context = try initFromClContext(allocator, cl_ctx);
    return context;
}

pub fn initFromBestDevice(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.Properties,
    device_type: cl.device.Type,
) !*Context {
    const platforms = try cl.platform.getAll(allocator);
    defer cl.platform.releaseList(allocator, platforms);

    var best_device: ?cl.device.DeviceId = null;
    var best_score: u64 = 0;
    for (platforms) |plat| {
        var num_devices: u32 = undefined;
        cl.device.getIds(plat.id.?, device_type, null, &num_devices) catch continue;
        if (num_devices == 0) continue;

        const devices = try allocator.alloc(cl.device.DeviceId, num_devices);
        defer {
            for (devices) |dev| {
                if (dev != best_device) {
                    cl.device.release(dev);
                }
            }
            allocator.free(devices);
        }

        try cl.device.getIds(plat.id.?, device_type, devices, null);
        var device_choosen_in_this_iteration: bool = false;
        for (devices) |device| {
            var max_work_group_size: u64 = undefined;
            try cl.device.getInfo(
                device,
                cl.device.Info.max_work_group_size,
                @sizeOf(u64),
                &max_work_group_size,
                null,
            );

            var compute_units: u32 = undefined;
            try cl.device.getInfo(
                device,
                cl.device.Info.partition_max_sub_devices,
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

    const context = try initFromClContext(allocator, cl_ctx);
    return context;
}

pub fn createOnePerPlatform(
    allocator: std.mem.Allocator,
    properties: ?[]const cl.context.Properties,
    device_type: cl.device.Type,
) !*Context {
    const platforms = try cl.platform.getAll(allocator);
    defer cl.platform.releaseList(allocator, platforms);

    const contexts = try allocator.alloc(*Context, platforms.len);
    var contexts_created: usize = 0;
    errdefer {
        for (contexts[0..contexts_created]) |ctx| {
            ctx.deinit();
        }
        allocator.free(contexts);
    }

    for (platforms, contexts) |*plat_details, *context| {
        var num_devices: u32 = undefined;
        cl.device.getIds(plat_details.id.?, device_type, null, &num_devices) catch continue;
        if (num_devices == 0) continue;

        const devices = try allocator.alloc(cl.device.DeviceId, num_devices);
        defer allocator.free(devices);

        try cl.device.getIds(plat_details.id.?, device_type, devices, null);

        context.* = init(allocator, properties, devices);
        contexts_created += 1;
    }

    return contexts;
}

pub fn initFromClContext(allocator: std.mem.Allocator, cl_ctx: cl.context.Context) !*Context {
    var context = try allocator.create(Context);
    errdefer allocator.destroy(context);

    var number_of_devices: u32 = undefined;
    try cl.context.getInfo(
        cl_ctx,
        cl.context.Info.num_devices,
        @sizeOf(u32),
        &number_of_devices,
        null,
    );

    const devices: []cl.device.DeviceId = try allocator.alloc(
        cl.device.DeviceId,
        @intCast(number_of_devices),
    );
    defer allocator.free(devices);

    try cl.context.getInfo(
        cl_ctx,
        cl.context.Info.devices,
        @sizeOf(cl.device.DeviceId) * number_of_devices,
        @ptrCast(devices.ptr),
        null,
    );

    context.allocator = allocator;
    context.ctx = cl_ctx;
    context.command_queues = try CommandQueue.initMultiples(allocator, context, devices);
    errdefer CommandQueue.deinitMultiples(allocator, context.command_queues);

    // context.events_batch_queue = events_releaser.EventsBatchQueue.init(allocator);
    // {
    //     errdefer context.events_batch_queue.release();

    //     context.events_releaser_thread = try std.Thread.spawn(
    //         .{},
    //         events_releaser.eventsBatchReleaserWorker,
    //         .{&context.events_batch_queue},
    //     );
    // }
    // errdefer {
    //     context.events_batch_queue.release();
    //     context.events_releaser_thread.join();
    // }

    return context;
}

pub fn deinit(context: *Context) void {
    const allocator = context.allocator;

    CommandQueue.deinitMultiples(allocator, context.command_queues);

    cl.context.release(context.ctx);

    // context.events_batch_queue.release();
    // context.events_releaser_thread.join();

    allocator.destroy(context);
}

pub fn deinitMultiples(allocator: std.mem.Allocator, contexts: []*Context) void {
    for (contexts) |ctx| {
        ctx.deinit();
    }
    allocator.free(contexts);
}

const Context = @This();
