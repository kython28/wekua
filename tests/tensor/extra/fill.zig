const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn check_elements(
    ctx: wekua.context.wContext,
    w_cmd: wekua.command_queue.wCommandQueue,
    tensor: wekua.tensor.wTensor
) !void {
    const cmd = w_cmd.cmd;

    const event_to_wait = wekua.tensor.event.acquire_tensor(tensor, .read);
    const custom_event = cl.event.create_user_event(ctx.ctx) catch |err| {
        tensor.mutex.unlock();
        return err;
    };
    defer cl.event.set_user_event_status(custom_event, .complete) catch unreachable;

    wekua.tensor.event.register_new_event(w_cmd, tensor, null, null, custom_event, .read) catch |err| {
        tensor.mutex.unlock();
        return err;
    };
    tensor.mutex.unlock();

    if (event_to_wait) |e| {
        try cl.event.wait(e);
    }

    var event_to_map: cl.event.cl_event = undefined;
    const map: []u64 = try cl.buffer.map(
        []u64, cmd, tensor.buffer, false,
        @intFromEnum(cl.buffer.enums.map_flags.read),
        0, tensor.size, null, &event_to_map
    );
    defer {
        var event_to_unmap: cl.event.cl_event = undefined;
        cl.buffer.unmap([]u64, cmd, tensor.buffer, map, null, &event_to_unmap) catch unreachable;
        cl.event.wait(event_to_unmap) catch unreachable;
        cl.event.release(event_to_unmap) catch unreachable;
        cl.event.release(event_to_map) catch unreachable;
    }

    try cl.event.wait(event_to_map);
    for (map) |elem| {
        try std.testing.expectEqual(64, elem);
    }
}

test "fill and check" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, .{.dtype = wekua.tensor.wTensorDtype.uint64});
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: u64 = 64;
    try wekua.tensor.extra.fill(w_cmd, tensor, &scalar, null);

    try check_elements(ctx, w_cmd, tensor);
}

test "fill multiple times and check" {
    const ctx = try wekua.context.create_from_device_type(&allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, .{.dtype = wekua.tensor.wTensorDtype.uint64});
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: u64 = 64;

    for (0..30) |s| {
        try wekua.tensor.extra.fill(w_cmd, tensor, &s, null);
    }
    try wekua.tensor.extra.fill(w_cmd, tensor, &scalar, null);

    try check_elements(ctx, w_cmd, tensor);
}
