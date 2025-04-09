pub const dot = @import("dot.zig");
pub const sum = @import("sum.zig");
pub const mean = @import("mean.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
