const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn check_elements(
    comptime T: type,
    ctx: *const wekua.core.Context,
    w_cmd: *const wekua.core.CommandQueue,
    tensor: *wekua.Tensor(T),
    expect_value_real: T,
    expect_value_imag: T
) !void {
    const cmd = w_cmd.cmd;

    const events_to_wait = tensor.events_manager.getPrevEvents(.read);
    const custom_event = try cl.event.create_user_event(ctx.ctx);
    {
        errdefer cl.event.release(custom_event);
        try tensor.events_manager.appendNewEvent(.read, events_to_wait, custom_event, null);
    }
    defer cl.event.set_user_event_status(custom_event, .complete) catch unreachable;

    if (events_to_wait) |events| {
        try cl.event.wait_for_many(events);
    }

    var event_to_map: cl.event.cl_event = undefined;
    const map: []T = try cl.buffer.map(
        []T, cmd, tensor.buffer, false,
        @intFromEnum(cl.buffer.enums.map_flags.read),
        0, tensor.size, null, &event_to_map
    );
    defer {
        var event_to_unmap: cl.event.cl_event = undefined;
        cl.buffer.unmap([]T, cmd, tensor.buffer, map, null, &event_to_unmap) catch unreachable;
        cl.event.wait(event_to_unmap) catch unreachable;
        cl.event.release(event_to_unmap);
        cl.event.release(event_to_map);
    }

    try cl.event.wait(event_to_map);
    if (tensor.is_complex){
        const n_elements = tensor.number_of_elements/2;
        for (0..n_elements) |i| {
            switch (@typeInfo(T)) {
                .int => {
                    try std.testing.expectEqual(expect_value_real, map[i*2]);
                    try std.testing.expectEqual(expect_value_imag, map[i*2 + 1]);
                },
                .float => {
                    try std.testing.expectApproxEqAbs(expect_value_real, map[i*2], comptime std.math.floatEps(T));
                    try std.testing.expectApproxEqAbs(expect_value_imag, map[i*2 + 1], comptime std.math.floatEps(T));
                },
                else => unreachable
            }
        }
    }else{
        for (map) |elem| {
            switch (@typeInfo(T)) {
                .int => try std.testing.expectEqual(expect_value_real, elem),
                .float => try std.testing.expectApproxEqAbs(expect_value_real, elem, comptime std.math.floatEps(T)),
                else => unreachable
            }
        }
    }
}

fn fill_and_check(comptime T: type, ctx: *const wekua.core.Context, tensor: *wekua.Tensor(T)) !void {
    const w_cmd = &ctx.command_queues[0];

    const value: T = switch (@typeInfo(T)) {
        .int => std.crypto.random.int(T),
        .float => std.crypto.random.float(T),
        else => unreachable
    };

    try wekua.tensor.fill.constant(T, tensor, w_cmd, value, null);

    try check_elements(T, ctx, w_cmd, tensor, value, 0);
}

fn fill_multiple_and_check(comptime T: type, ctx: *const wekua.core.Context, tensor: *wekua.Tensor(T)) !void {
    const w_cmd = &ctx.command_queues[0];
    const scalar = switch (@typeInfo(T)) {
        .int => std.crypto.random.int(T),
        .float => std.crypto.random.float(T),
        else => unreachable
    };

    for (0..30) |_| {
        const s = switch (@typeInfo(T)) {
            .int => std.crypto.random.int(T),
            .float => std.crypto.random.float(T),
            else => unreachable
        };

        try wekua.tensor.fill.constant(T, tensor, w_cmd, s, null);
    }
    try wekua.tensor.fill.constant(T, tensor, w_cmd, scalar, null);

    try check_elements(T, ctx, w_cmd, tensor, scalar, 0);
}

fn fill_multiple_and_check2(comptime T: type, ctx: *const wekua.core.Context, tensor: *wekua.Tensor(T)) !void {
    const w_cmd = &ctx.command_queues[0];
    const scalar = switch (@typeInfo(T)) {
        .int => std.crypto.random.int(T),
        .float => std.crypto.random.float(T),
        else => unreachable
    };

    for (0..30) |_| {
        const s = switch (@typeInfo(T)) {
            .int => std.crypto.random.int(T),
            .float => std.crypto.random.float(T),
            else => unreachable
        };

        try wekua.tensor.fill.constant(T, tensor, w_cmd, s, null);
        try check_elements(T, ctx, w_cmd, tensor, s, 0);
    }
    try wekua.tensor.fill.constant(T, tensor, w_cmd, scalar, null);

    try check_elements(T, ctx, w_cmd, tensor, scalar, 0);
}

fn fill_and_check2(comptime T: type, ctx: *const wekua.core.Context, tensor: *wekua.Tensor(T)) !void {
    const w_cmd = &ctx.command_queues[0];

    try wekua.tensor.fill.one(T, tensor, w_cmd);

    try check_elements(T, ctx, w_cmd, tensor, 1, 0);
}

fn fill_and_check3(comptime T: type, ctx: *const wekua.core.Context, tensor: *wekua.Tensor(T)) !void {
    const w_cmd = &ctx.command_queues[0];

    try wekua.tensor.fill.zeroes(T, tensor, w_cmd);

    try check_elements(T, ctx, w_cmd, tensor, 0, 0);
}

fn fill_complex_and_check(comptime T: type, ctx: *const wekua.core.Context, tensor: *wekua.Tensor(T)) !void {
    const w_cmd = &ctx.command_queues[0];

    const value: T = switch (@typeInfo(T)) {
        .int => std.crypto.random.int(T),
        .float => std.crypto.random.float(T),
        else => unreachable
    };

    const value2: T = switch (@typeInfo(T)) {
        .int => std.crypto.random.int(T),
        .float => std.crypto.random.float(T),
        else => unreachable
    };

    try wekua.tensor.fill.constant(T, tensor, w_cmd, value, value2);

    try check_elements(T, ctx, w_cmd, tensor, value, value2);
}

test "fill and check" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    inline for (wekua.core.SupportedTypes) |T| {
        const tensor = try wekua.Tensor(T).alloc(ctx, &[_]u64{20, 10}, .{});
        defer tensor.release();

        try fill_and_check(T, ctx, tensor);
        try fill_and_check2(T, ctx, tensor);
        try fill_and_check3(T, ctx, tensor);
        try fill_multiple_and_check(T, ctx, tensor);
        try fill_multiple_and_check2(T, ctx, tensor);

        const tensor2 = try wekua.Tensor(T).alloc(ctx, &[_]u64{20, 10}, .{
            .is_complex = true
        });
        defer tensor2.release();

        try fill_complex_and_check(T, ctx, tensor2);
    }
}
