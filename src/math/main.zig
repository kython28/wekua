pub const trig = @import("trig.zig");
pub const basic = @import("basic.zig");

pub const sin = trig.sin;
pub const cos = trig.cos;
pub const tan = trig.tan;
pub const sinh = trig.sinh;
pub const cosh = trig.cosh;
pub const tanh = trig.tanh;

pub const dot = basic.dot;
pub const sum = basic.sum;
pub const mean = basic.mean;

test {
    _ = trig;
    _ = basic;
}
