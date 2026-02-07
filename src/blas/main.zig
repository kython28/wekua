const axpy_module = @import("axpy.zig");
const gemm_module = @import("gemm.zig");

pub const axpy = axpy_module.axpy;
pub const gemm = gemm_module.gemm;
pub const GemmOperation = gemm_module.Operation;

test {
    _ = axpy_module;
    _ = gemm_module;
}
