const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");


fn test_convertions_to_complex(
    allocator: std.mem.Allocator, ctx: wekua.context.wContext, randprg: std.Random,
    comptime T: type, comptime dom: bool
) !void {
    var shape: [4]u64 = .{
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20)
    };
    const tensor = try wekua.tensor.alloc(ctx, &shape, .{
        .dtype = wekua.tensor.get_wekua_dtype_from_zig_type(T),
        .is_complex = false
    });
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.random.random(w_cmd, tensor);

    const tensor2 = try wekua.tensor.alloc(ctx, &shape, .{
        .dtype = wekua.tensor.get_wekua_dtype_from_zig_type(T),
        .is_complex = true
    });
    defer wekua.tensor.release(tensor2);

    try wekua.tensor.extra.convertions.to_complex(w_cmd, tensor, tensor2, dom);

    const numbers1: []T = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers2: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    try wekua.tensor.io.write_to_buffer(w_cmd, tensor, numbers1);
    try wekua.tensor.io.write_to_buffer(w_cmd, tensor2, numbers2);
    for (numbers1, 0..) |n1, index| {
        const n2 = numbers2[index*2 + @intFromBool(dom)];
        const n3 = numbers2[index*2 + (1 - @intFromBool(dom))];
        if (T == f32 or T == f64) {
            const eps = std.math.floatEps(T);
            try std.testing.expectApproxEqAbs(n1, n2, eps);
            try std.testing.expectApproxEqAbs(0.0, n3, eps);
        }else{
            try std.testing.expectEqual(n1, n2);
            try std.testing.expectEqual(0, n3);
        }
    }
}

test "Convert tensor to complex" {
    const allocator = std.testing.allocator;
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const types = .{u8, i8, u16, i16, u32, i32, u64, i64, f32, f64};
    const randprg = std.crypto.random;
    inline for (types) |T| {
        try test_convertions_to_complex(allocator, ctx, randprg, T, false);
        try test_convertions_to_complex(allocator, ctx, randprg, T, true);
    }
}
