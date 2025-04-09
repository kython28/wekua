pub const basics = @import("basics/main.zig");
pub const trig = @import("trig.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
