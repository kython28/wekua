const std = @import("std");

const w_errors = @import("errors.zig");

const dtypes = @import("dtypes.zig");
const wTensor = dtypes.wTensor;

pub inline fn eql_tensors_dimensions(tensor_a: wTensor, tensor_b: wTensor) !void {
    if (tensor_a.dtype != tensor_b.dtype) return w_errors.errors.UnqualTensors;
    if (!std.mem.eql(u64, tensor_a.shape, tensor_b.shape)) return w_errors.errors.UnqualTensors;
    if (tensor_a.is_complex != tensor_b.is_complex) return w_errors.errors.UnqualTensors;
}
