const std = @import("std");
const cl = @import("opencl");
const command_queue = @import("command_queue.zig");

const _wcontext = struct {
    allocator: std.mem.Allocator,
    ctx: cl.context.cl_context,
    command_queues: []command_queue.wCommandQueue
};

pub const wContext = *_wcontext;

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
    // context.command_queues_status = comptime blk: {
    //     const max_value: u64 = @intCast(std.math.pow(u65, 2, 64) - 1);
    //     break :blk max_value;
    // };

    return context;
}

pub fn release(context: wContext) void {
    const allocator = context.allocator;

    cl.context.release(context.ctx) catch unreachable;
    command_queue.release_multiple(allocator, context.command_queues);
    allocator.destroy(context);
}

