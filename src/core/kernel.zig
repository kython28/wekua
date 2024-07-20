const std = @import("std");
const cl = @import("opencl");

pub const wCommandQueue = @import("command_queue.zig").wCommandQueue;
pub const wTensorDtype = @import("../tensor/utils/dtypes.zig").wTensorDtype;

pub const wKernelsID = enum (u16) {
    Random = 0,
    RandRange = 1,
};

pub const total_number_of_kernels: u16 = @intCast(@typeInfo(wKernelsID).Enum.fields.len);

const _w_kernel = struct {
    kernels: ?[]?cl.kernel.cl_kernel,
    programs: ?[]?cl.program.cl_program
};

pub const wKernel = *_w_kernel;

pub const wCompileOptions = struct {
    dtype: wTensorDtype,
    is_complex: bool = false,
    vectors_enabled: bool = true,
    kernel_name: []const u8,
    extra_args: ?[]const u8
};

pub fn compile_kernel(
    command_queue: wCommandQueue,
    options: wCompileOptions,
    kernel: *cl.kernel.cl_kernel,
    program: *cl.program.cl_program,
    content: []const u8
) !void {
    const cl_ctx = command_queue.ctx;
    program.* = try cl.program.create_with_source(
        cl_ctx, 1, &@as([*][]const u8, @ptrCast(content))[0..1],
        command_queue.allocator.*
    );
    errdefer cl.program.release(program.*) catch unreachable;

    const allocator = command_queue.allocator;
    var args: []u8 = undefined;

    const dtype = options.dtype;
    const vector_width = if (options.vectors_enabled and !options.is_complex) command_queue.vector_widths[@intFromEnum(dtype)] else 1;
    if (options.extra_args) |v| {
        args = try std.fmt.allocPrint(
            allocator.*, "-Dwk_width={d} -Ddtype={d} -Dmem_type={d} -Dcom={d} {}",
            .{
                vector_width,
                @intFromEnum(dtype),
                command_queue.local_mem_type,
                @intFromBool(options.is_complex),
                v
            }
        );
    }else{
        args = try std.fmt.allocPrint(
            allocator.*, "-Dwk_width={d} -Ddtype={d} -Dmem_type={d} -Dcom={d}",
            .{
                vector_width,
                @intFromEnum(dtype),
                command_queue.local_mem_type,
                @intFromBool(options.is_complex)
            }
        );
    }
    defer allocator.free(args);

    const ret = cl.program.build(
        program.*, 1, @as([*]cl.device.cl_device_id, @ptrCast(command_queue.device))[0..1],
        args, null, null
    );

    if (ret) |_| {
        kernel.* = try cl.kernel.create(program.*, options.kernel_name);
    }else |err| {
        switch (err) {
            error.build_program_failure => {
                var msg_len: usize = undefined;
                try cl.program.get_build_info(program.*, command_queue.device, .build_log, 0, null, &msg_len);

                const compile_log = try allocator.alloc(u8, msg_len);
                defer allocator.free(compile_log);

                try cl.program.get_build_info(program.*, command_queue.device, .build_log, msg_len, compile_log, null);

                std.log.warn("{}", .{compile_log});
            },
            else => {}
        }
        return err;
    }
}

pub fn get_kernel_from_id(command_queue: wCommandQueue, kernel_id: wKernelsID) !wKernel {
    var kernel = command_queue.kernels[@intFromEnum(kernel_id)];
    if (kernel) |v| {
        return v;
    }

    kernel = try command_queue.allocator.create(_w_kernel);
    kernel.?.kernels = null;
    kernel.?.programs = null;
    return kernel;
}

pub fn get_kernel(
    command_queue: wCommandQueue, kernel_id: wKernelsID, number_of_cl_kernels: usize
) !wKernel {
    const kernel = try get_kernel_from_id(command_queue, kernel_id);

    if (kernel.kernels == null) {
        const allocator = command_queue.allocator;

        // It is 4 because: float32, float64, complex_float32 and complex_float64
        const kernels = try allocator.alloc(?cl.kernel.cl_kernel, number_of_cl_kernels);
        errdefer allocator.free(kernels);
        @memset(kernels, null);

        const programs = try allocator.alloc(?cl.program.cl_program, number_of_cl_kernels);
        @memset(programs, null);

        kernel.kernels = kernels;
        kernel.programs = programs;
    }

    return kernel;
}
