const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn check_elements(
    comptime T: type,
    ctx: wekua.Context,
    w_cmd: wekua.core.CommandQueue,
    tensor: wekua.Tensor(T),
    expect_value_real: T,
    expect_value_imag: T
) !void {
    const cmd = w_cmd.cmd;

    const events_to_wait = tensor.events_manager.getPrevEvents(.read);
    const custom_event = cl.event.create_user_event(ctx.ctx);
    defer cl.event.set_user_event_status(custom_event, .complete) catch unreachable;

    tensor.events_manager.appendNewEvent(.read, events_to_wait, custom_event, null, true);
    wekua.tensor.event.register_new_event_to_single_tensor(w_cmd, tensor, null, null, custom_event, .read);
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

// test "fill and check" {
fn fill_and_check(comptime T: type, ctx: *const wekua.core.Context, tensor: wekua.Tensor(T)) !void {
    const w_cmd = ctx.command_queues[0];

    const value: T = switch (@typeInfo(T)) {
        .int => std.crypto.random.int(T),
        .float => std.crypto.random.float(T),
        else => unreachable
    };

    try wekua.tensor.fill.constant(T, tensor, w_cmd, value, null);

    try check_elements(T, ctx, w_cmd, tensor, value, 0);
}

fn fill_multiple_and_check(_: std.mem.Allocator, ctx: wekua.context.wContext, tensor: wekua.tensor.wTensor) !void {
    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};

    for (0..30) |s| {
        try wekua.tensor.extra.fill(w_cmd, tensor, .{ .uint64 = s }, null);
    }
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, null);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, 0);
}

fn fill_multiple_and_check2(_: std.mem.Allocator, ctx: wekua.context.wContext, tensor: wekua.tensor.wTensor) !void {
    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};

    for (0..30) |s| {
        try wekua.tensor.extra.fill(w_cmd, tensor, .{ .uint64 = s}, null);
        try check_elements(ctx, w_cmd, tensor, s, 0);
    }
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, null);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, 0);
}

fn fill_complex_and_check(_: std.mem.Allocator, ctx: wekua.context.wContext, tensor: wekua.tensor.wTensor) !void {
    const w_cmd = ctx.command_queues[0];
    const scalar: wekua.tensor.wScalar = .{.uint64 = 64};
    const imag: wekua.tensor.wScalar = .{.uint64 = 4234234};
    try wekua.tensor.extra.fill(w_cmd, tensor, scalar, imag);

    try check_elements(ctx, w_cmd, tensor, scalar.uint64, imag.uint64);
}

fn fill_complex_multiple_and_check(_: std.mem.Allocator, ctx: wekua.context.wContext, tensor: wekua.tensor.wTensor) !void {
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

fn fill_complex_multiple_and_check2(_: std.mem.Allocator, ctx: wekua.context.wContext, tensor: wekua.tensor.wTensor) !void {
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

test "fill and check" {
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    inline for (wekua.tensor.SupportedTypes) |T| {
        const tensor = try wekua.Tensor(T).alloc(ctx, &[_]u64{20, 10}, .{});
        defer tensor.release();

        try fill_and_check(allocator, ctx, tensor);
        try fill_multiple_and_check(allocator, ctx, tensor);
        try fill_multiple_and_check2(allocator, ctx, tensor);
    }


//     try std.testing.checkAllAllocationFailures(allocator, fill_and_check, .{ctx, tensor});
//     try std.testing.checkAllAllocationFailures(allocator, fill_multiple_and_check, .{ctx, tensor});
//     try std.testing.checkAllAllocationFailures(allocator, fill_multiple_and_check2, .{ctx, tensor});
// }

// test "fill complex and check" {
//     const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
//     defer wekua.context.release(ctx);

//     const config: wekua.tensor.wCreateTensorConfig = .{
//         .dtype = wekua.tensor.wTensorDtype.uint64,
//         .is_complex = true
//     };
//     const tensor = try wekua.tensor.alloc(ctx, &[_]u64{20, 10}, config);
//     defer wekua.tensor.release(tensor);

//     try fill_complex_and_check(allocator, ctx, tensor);
//     try fill_complex_multiple_and_check(allocator, ctx, tensor);
//     try fill_complex_multiple_and_check2(allocator, ctx, tensor);

//     try std.testing.checkAllAllocationFailures(allocator, fill_complex_and_check, .{ctx, tensor});
//     try std.testing.checkAllAllocationFailures(allocator, fill_complex_multiple_and_check, .{ctx, tensor});
//     try std.testing.checkAllAllocationFailures(allocator, fill_complex_multiple_and_check2, .{ctx, tensor});
}
