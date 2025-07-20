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

// Unit Tests
const testing = std.testing;

test "getTypeId returns correct indices for supported types" {
    try testing.expectEqual(@as(comptime_int, 0), getTypeId(i8));
    try testing.expectEqual(@as(comptime_int, 1), getTypeId(u8));
    try testing.expectEqual(@as(comptime_int, 2), getTypeId(i16));
    try testing.expectEqual(@as(comptime_int, 3), getTypeId(u16));
    try testing.expectEqual(@as(comptime_int, 4), getTypeId(i32));
    try testing.expectEqual(@as(comptime_int, 5), getTypeId(u32));
    try testing.expectEqual(@as(comptime_int, 6), getTypeId(i64));
    try testing.expectEqual(@as(comptime_int, 7), getTypeId(u64));
    try testing.expectEqual(@as(comptime_int, 8), getTypeId(f32));
    try testing.expectEqual(@as(comptime_int, 9), getTypeId(f64));
}

test "SupportedTypes array contains expected types" {
    try testing.expectEqual(@as(usize, 10), SupportedTypes.len);
    try testing.expectEqual(i8, SupportedTypes[0]);
    try testing.expectEqual(u8, SupportedTypes[1]);
    try testing.expectEqual(i16, SupportedTypes[2]);
    try testing.expectEqual(u16, SupportedTypes[3]);
    try testing.expectEqual(i32, SupportedTypes[4]);
    try testing.expectEqual(u32, SupportedTypes[5]);
    try testing.expectEqual(i64, SupportedTypes[6]);
    try testing.expectEqual(u64, SupportedTypes[7]);
    try testing.expectEqual(f32, SupportedTypes[8]);
    try testing.expectEqual(f64, SupportedTypes[9]);
}

test "Context struct has expected fields" {
    // Test that Context struct has the expected fields
    const context_info = @typeInfo(Context);
    try testing.expect(context_info == .Struct);
    
    // Verify field names exist (compile-time check)
    const has_allocator = @hasField(Context, "allocator");
    const has_ctx = @hasField(Context, "ctx");
    const has_command_queues = @hasField(Context, "command_queues");
    
    try testing.expect(has_allocator);
    try testing.expect(has_ctx);
    try testing.expect(has_command_queues);
}

test "getTypeId compile error for unsupported types" {
    // This test verifies that unsupported types cause compile errors
    // We can't actually test the compile error in a unit test,
    // but we can document the expected behavior
    
    // The following would cause compile errors:
    // _ = getTypeId(bool); // Should fail
    // _ = getTypeId([]u8); // Should fail
    // _ = getTypeId(struct{}); // Should fail
    
    // Instead, we test that all supported types work
    comptime {
        _ = getTypeId(i8);
        _ = getTypeId(u8);
        _ = getTypeId(i16);
        _ = getTypeId(u16);
        _ = getTypeId(i32);
        _ = getTypeId(u32);
        _ = getTypeId(i64);
        _ = getTypeId(u64);
        _ = getTypeId(f32);
        _ = getTypeId(f64);
    }
}
