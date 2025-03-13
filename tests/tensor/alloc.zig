const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

fn create_and_release(
    comptime T: type,
    ctx: *const wekua.core.Context,
    config: wekua.CreateTensorConfig,
) !void {
    const tensor = try wekua.Tensor(T).alloc(ctx, &[_]u64{ 20, 10 }, config);
    tensor.release();
}

fn create_check_and_release(comptime T: type, ctx: *const wekua.core.Context, config: wekua.CreateTensorConfig) !void {
    const tensor = try wekua.Tensor(T).alloc(ctx, &[_]u64{ 20, 10 }, config);
    defer tensor.release();

    const w_cmd = ctx.command_queues[0];
    const cmd = w_cmd.cmd;

    const events_to_wait = tensor.events_manager.getPrevEvents(.read);
    const custom_event = try cl.event.create_user_event(ctx.ctx);
    defer cl.event.set_user_event_status(custom_event, .complete) catch unreachable;

    try tensor.events_manager.appendNewEvent(.read, events_to_wait, custom_event, null, true);

    var event_to_map: cl.event.cl_event = undefined;
    const map: []u8 = try cl.buffer.map(
        []u8,
        cmd,
        tensor.buffer,
        false,
        @intFromEnum(cl.buffer.enums.map_flags.read),
        0,
        tensor.size,
        events_to_wait,
        &event_to_map,
    );
    defer {
        var event_to_unmap: cl.event.cl_event = undefined;
        cl.buffer.unmap([]u8, cmd, tensor.buffer, map, null, &event_to_unmap) catch unreachable;
        cl.event.wait(event_to_unmap) catch unreachable;
        cl.event.release(event_to_unmap) catch unreachable;
        cl.event.release(event_to_map) catch unreachable;
    }

    try cl.event.wait(event_to_map);
    for (map) |elem| {
        try std.testing.expectEqual(0, elem);
    }
}

test "create and release" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    inline for (wekua.tensor.SupportedTypes) |T| {
        try create_and_release(T, ctx, .{});

        try create_and_release(T, ctx, .{ .is_complex = true });
    }
}

test "create, check and release" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    inline for (wekua.tensor.SupportedTypes) |T| {
        try create_check_and_release(T, ctx, .{});

        try create_check_and_release(T, ctx, .{ .is_complex = true });
    }
}
