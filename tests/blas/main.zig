pub const axpy = @import("axpy.zig");
pub const gemm = @import("gemm.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}

