const std = @import("std");
const cl = @import("opencl");

const wekua = @import("wekua");

test "create" {
    const ally = std.testing.allocator;

    const ctx = wekua.context.create_from_device_type(&ally, null, cl.device.enums.device_type.all);
    if (ctx) |v| {
        wekua.context.release(v);
    }else |err| {
        std.debug.print("Error: {any}\n", .{err});
        return err;
    }
}

test "create_with_fail" {
    const max_index = 10;
    for (0..max_index) |i| {
        var failing_allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{
            .fail_index = i
        });
        const ally = failing_allocator.allocator();

        _ = wekua.context.create_from_device_type(
            &ally, null, cl.device.enums.device_type.all
        ) catch |err| switch (err) {
            error.OutOfMemory => return,
            else => return err
        };
    }
}
