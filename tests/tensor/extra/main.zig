pub const fill = @import("fill.zig");
pub const memory = @import("memory.zig");
pub const random = @import("random.zig");
pub const transpose = @import("transpose.zig");
// pub const convertions = @import("convertions.zig");
// pub const identity = @import("identity.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
