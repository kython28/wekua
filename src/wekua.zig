pub const opencl = @import("opencl");

pub const core = @import("core/main.zig");

pub const tensor = @import("tensor/main.zig");
pub const Tensor = tensor.Tensor;
pub const TensorErrors = tensor.Errors;
pub const CreateTensorConfig = tensor.CreateConfig;


pub const utils = @import("utils/utils.zig");

pub const blas = @import("blas/main.zig");
pub const math = @import("math/main.zig");

pub const nn = @import("nn/main.zig");
