pub const w_empty = @import("empty.zig");
pub const w_alloc = @import("alloc.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
