pub const context = @import("core/context.zig");

// pub const utils = @import("utils/utils.zig"); // TODO
// pub const tensor = @import("tensor/main.zig");

// pub const blas = @import("blas/main.zig");
// pub const math = @import("math/main.zig");
// pub const nn = @import("nn/main.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
