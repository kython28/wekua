pub const axpy = @import("axpy.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}

