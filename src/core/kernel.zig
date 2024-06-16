const cl = @import("opencl");

pub const wKernelsID = enum (u16) {
    Random = 0,
    RandRange = 1,
};

pub const total_number_of_kernels: u16 = @intCast(@typeInfo(wKernelsID).Enum.fields.len);

const _w_kernel = struct {
    kernel: []cl.kernel.cl_kernel,
    program: []cl.program.cl_program
};

pub const wKernel = *_w_kernel;
