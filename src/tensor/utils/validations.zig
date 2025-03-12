const std = @import("std");

const w_tensor = @import("../main.zig");
const wTensor = w_tensor.wTensor;

pub inline fn eql_number_space(comptime T: type, tensor_a: *wTensor(T), tensor_b: *wTensor(T)) !void {
    if (tensor_a.is_complex != tensor_b.is_complex) return w_tensor.wTensorErrors.TensorDoesNotSupportComplexNumbers;
}

pub inline fn eql_tensors_dimensions(comptime T: type, tensor_a: *wTensor(T), tensor_b: *wTensor(T)) !void {
    if (!std.mem.eql(u64, tensor_a.shape, tensor_b.shape)) return w_tensor.wTensorErrors.UnqualTensorsShape;
}

pub inline fn eql_tensors(comptime T: type, tensor_a: *wTensor(T), tensor_b: *wTensor(T)) !void {
    try eql_tensors_dimensions(T, tensor_a, tensor_b);
    try eql_number_space(T, tensor_a, tensor_b);
    if (tensor_a.vectors_enabled != tensor_b.vectors_enabled) return w_tensor.wTensorErrors.UnqualTensorsAttribute;
}
