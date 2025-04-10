const std = @import("std");
const wekua = @import("wekua");
const cl = @import("opencl");

const core = wekua.core;
const CommandQueue = core.CommandQueue;

const w_tensor = @import("main.zig");
const Tensor = w_tensor.Tensor;

// TODO: Write tests for checking the output of print
//
//
//
test "Tensor print - semantic analysis check" {
    const ctx = try wekua.core.Context.init_from_device_type(
        std.testing.allocator,
        null,
        cl.device.enums.device_type.all,
    );
    defer ctx.release();

    const cmd = &ctx.command_queues[0];

    inline for (wekua.core.SupportedTypes) |T| {
        if (cmd.typeIsSupported(T)) {
            inline for (&.{ false, true }) |is_complex| {
                const tensor = try wekua.Tensor(T).alloc(ctx, &.{ 10, 10 }, .{ .is_complex = is_complex });
                defer tensor.release();

                try wekua.tensor.printZ(T, std.io.null_writer, cmd, tensor);
            }
        }
    }
}
