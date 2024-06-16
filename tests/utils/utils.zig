pub const linked_list = @import("linked_list.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
