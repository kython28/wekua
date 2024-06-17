const wTensor = @import("../empty.zig").wTensor;

pub fn fill(tensor: wTensor, real_scalar: ?*anyopaque, imag_scalar: ?*anyopaque) !void {
    const dtype = tensor.dtype;
    var pattern_size: usize = 0;
    if (real_scalar != null) {

    }
}
