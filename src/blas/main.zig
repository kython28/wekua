const axpy_module = @import("axpy.zig");

pub const axpy = axpy_module.axpy;
// pub const gemm = @import("gemm.zig").gemm;

test {
    _ = axpy_module;
}
