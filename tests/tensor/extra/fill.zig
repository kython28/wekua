const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn check_elements(
    ctx: wekua.context.wContext,
    w_cmd: wekua.command_queue.wCommandQueue,
    tensor: wekua.tensor.wTensor,
    expect_value_real: u64,
    expect_value_imag: u64
) !void {
    const cmd = w_cmd.cmd;

    const events_to_wait = wekua.tensor.event.acquire_tensor(tensor, .read);
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

    if (events_to_wait) |e| {
        try cl.event.wait_for_many(e);
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
    if (tensor.is_complex){
        const n_elements = tensor.number_of_elements/2;
        for (0..n_elements) |i| {
            try std.testing.expectEqual(expect_value_real, map[i*2]);
            try std.testing.expectEqual(expect_value_imag, map[i*2 + 1]);
        }
    }else{
        for (map) |elem| {
            try std.testing.expectEqual(expect_value_real, elem);
        }
    }
}

test "try to fill with exceptions" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, .{.dtype = wekua.tensor.wTensorDtype.uint64});
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    try std.testing.expectError(
        wekua.tensor.errors.InvalidScalarDtype,
        wekua.tensor.extra.fill(w_cmd, tensor, .{ .uint32 = 1 }, null)
    );
    try std.testing.expectError(
        wekua.tensor.errors.TensorIsnotComplex,
        wekua.tensor.extra.fill(w_cmd, tensor, null, .{ .uint8 = 19 })
    );
}

test "fill and check" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, .{.dtype = wekua.tensor.wTensorDtype.uint64});
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, null);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, 0);
}

test "fill multiple times and check" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, .{.dtype = wekua.tensor.wTensorDtype.uint64});
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};

    for (0..30) |s| {
        try wekua.tensor.extra.fill(w_cmd, tensor, .{ .uint64 = s }, null);
    }
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, null);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, 0);
}

test "fill multiple times and check2" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, .{.dtype = wekua.tensor.wTensorDtype.uint64});
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};

    for (0..30) |s| {
        try wekua.tensor.extra.fill(w_cmd, tensor, .{ .uint64 = s}, null);
        try check_elements(ctx, w_cmd, tensor, s, 0);
    }
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, null);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, 0);
}

test "fill complex and check" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const config: wekua.tensor.wCreateTensorConfig = .{
        .dtype = wekua.tensor.wTensorDtype.uint64,
        .is_complex = true
    };
    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, config);
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};
    const imag: wekua.tensor.wScalar = .{.uint64 = 4234234};
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, imag);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, imag.uint64);
}

test "fill complex multiple times and check" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const config: wekua.tensor.wCreateTensorConfig = .{
        .dtype = wekua.tensor.wTensorDtype.uint64,
        .is_complex = true
    };
    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, config);
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};
    const imag: wekua.tensor.wScalar = .{.uint64 = 4234234};

    for (1..30) |s| {
        const p: wekua.tensor.wScalar = .{.uint64 = s * 10 - 10};
        try wekua.tensor.extra.fill(w_cmd, tensor, .{ .uint64 = s}, p);
    }
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, imag);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, imag.uint64);
}

test "fill complex multiple times and check2" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const config: wekua.tensor.wCreateTensorConfig = .{
        .dtype = wekua.tensor.wTensorDtype.uint64,
        .is_complex = true
    };
    const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, config);
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};
    const imag: wekua.tensor.wScalar = .{.uint64 = 4234234};

    for (1..30) |s| {
        const p: wekua.tensor.wScalar = .{.uint64 = s * 10 - 10};
        try wekua.tensor.extra.fill(w_cmd, tensor, .{ .uint64 = s}, p);
        try check_elements(ctx, w_cmd, tensor, s, p.uint64);
    }
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, imag);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, imag.uint64);
}
