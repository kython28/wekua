const std = @import("std");
const cl = @import("opencl");

const wCommandQueue = @import("command_queue.zig").wCommandQueue;
const dtypes = @import("../tensor/utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wTensorDtype = dtypes.wTensorDtype;

const header_content: []const u8 = @embedFile("wekua_cl_lib.h");

pub const wKernelsID = enum (u16) {
    Random = 0,
    RandRange = 1,
    Transpose = 2,
    ToComplex = 3,
    ToReal = 4,

    AXPY = 5,
};

pub const total_number_of_kernels: u16 = @intCast(@typeInfo(wKernelsID).Enum.fields.len);
pub const total_number_of_headers: u16 = @as(u16, @intCast(@typeInfo(wTensorDtype).Enum.fields.len)) * 2 * 2;

pub const _w_kernel = struct {
    kernels: ?[]?cl.kernel.cl_kernel,
    programs: ?[]?cl.program.cl_program
};

pub const wKernel = *_w_kernel;

pub const wCompileOptions = struct {
    dtype: wTensorDtype,
    is_complex: bool = false,
    vectors_enabled: bool = true,
    kernel_name: []const u8,
    extra_args: ?[]const u8 = null
};

fn compile_header(
    command_queue: wCommandQueue, dtype: wTensorDtype, vectors_enabled: bool,
    is_complex: bool, _: []const u8
) !cl.program.cl_program {
    const headers = command_queue.headers;
    const index: u16 = @intFromEnum(dtype) * 4 + @as(u16, @intFromBool(vectors_enabled)) * 2 + @intFromBool(is_complex);

    if (headers.programs.?[index]) |prg| {
        return prg;
    }

    // const content: []const u8 = header_content; // Don't ask, this just work and i don't know why
    const new_header_prg = try cl.program.create_with_source(
        command_queue.ctx.ctx, @as([*]const []const u8, @ptrCast(&header_content))[0..1],
        command_queue.allocator
    );

    headers.programs.?[index] = new_header_prg;
    return new_header_prg;
}

fn show_build_log(program: cl.program.cl_program, command_queue: wCommandQueue) !void {
    var msg_len: usize = undefined;
    try cl.program.get_build_info(
        program, command_queue.device, .build_log, 0, null, &msg_len
    );

    const compile_log: []u8 = try command_queue.allocator.alloc(u8, msg_len);
    defer command_queue.allocator.free(compile_log);

    try cl.program.get_build_info(
        program, command_queue.device, .build_log, msg_len, compile_log.ptr, null
    );

    std.log.warn("{s}", .{compile_log});
}

pub fn compile_kernel(
    command_queue: wCommandQueue,
    options: wCompileOptions,
    kernel: *cl.kernel.cl_kernel,
    program: *cl.program.cl_program,
    content: []const u8
) !void {
    const cl_ctx = command_queue.ctx.ctx;
    const allocator = command_queue.allocator;
    const new_program = try cl.program.create_with_source(
        cl_ctx, @as([*]const []const u8, @ptrCast(&content))[0..1],
        allocator
    );
    defer cl.program.release(new_program) catch unreachable;

    var args: []u8 = undefined;

    const dtype = options.dtype;
    const vector_width = if (options.vectors_enabled and !options.is_complex) command_queue.vector_widths[@intFromEnum(dtype)] else 1;
    if (options.extra_args) |v| {
        args = try std.fmt.allocPrint(
            allocator, "-Dwk_width={d} -Ddtype={d} -Dmem_type={d} -Dcom={d} {s}\x00",
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
            allocator, "-Dwk_width={d} -Ddtype={d} -Dmem_type={d} -Dcom={d}\x00",
            .{
                vector_width,
                @intFromEnum(dtype),
                command_queue.local_mem_type,
                @intFromBool(options.is_complex)
            }
        );
    }
    defer allocator.free(args);

    const header_prg = try compile_header(command_queue, dtype, options.vectors_enabled, options.is_complex, args);
    const header_name: []const u8 = "wekua.h";

    const devices: []const cl.device.cl_device_id = @as([*]const cl.device.cl_device_id, @ptrCast(&command_queue.device))[0..1];
    cl.program.compile(
        allocator, new_program, devices, args,
        @as([*]const cl.program.cl_program, @ptrCast(&header_prg))[0..1],
        @as([*]const []const u8, @ptrCast(&header_name))[0..1], null, null
    ) catch |err| {
        switch (err) {
            error.compile_program_failure => {
                show_build_log(new_program, command_queue) catch unreachable;
            },
            else => {}
        }
        return err;
    };

    program.* = cl.program.link(
        cl_ctx, devices, null, @as([*]const cl.program.cl_program, @ptrCast(&new_program))[0..1],
        null, null
    ) catch |err| {
        switch (err) {
            error.link_program_failure => {
                show_build_log(new_program, command_queue) catch unreachable;
            },
            else => {}
        }
        return err;
    };

    kernel.* = try cl.kernel.create(program.*, options.kernel_name);
}

pub fn get_kernel_from_id(command_queue: wCommandQueue, kernel_id: wKernelsID) !wKernel {
    var kernel = command_queue.kernels[@intFromEnum(kernel_id)];
    if (kernel) |v| {
        return v;
    }

    kernel = try command_queue.allocator.create(_w_kernel);
    command_queue.kernels[@intFromEnum(kernel_id)] = kernel;
    kernel.?.kernels = null;
    kernel.?.programs = null;
    return kernel.?;
}

pub fn get_kernel(
    command_queue: wCommandQueue, kernel_id: wKernelsID, number_of_cl_kernels: usize
) !wKernel {
    const kernel = try get_kernel_from_id(command_queue, kernel_id);

    if (kernel.kernels == null) {
        const allocator = command_queue.allocator;

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

pub fn create_and_get_kernel(
    command_queue: wCommandQueue, kernel_id: wKernelsID, kernel_source: []const u8, options: wCompileOptions,
    comptime can_use_complex: bool, comptime can_use_vectors: bool, number_of_cl_kernels: usize,
    getting_index_func: anytype, extra_args: anytype
) !cl.kernel.cl_kernel {
    if (!can_use_complex and options.is_complex) {
        @panic("Kernels with complex numbers are not allowed");
    }
    if (!can_use_vectors and options.vectors_enabled) {
        @panic("Kernels with vectors are not allowed");
    }

    const kernels_set = try get_kernel(command_queue, kernel_id, number_of_cl_kernels);
    const index: usize = getting_index_func(options.is_complex, options.vectors_enabled, options.dtype, extra_args);
    if (kernels_set.kernels.?[index]) |kernel| {
        return kernel;
    }

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    try compile_kernel(
        command_queue, options,
        &kernel, &program,
        kernel_source
    );

    kernels_set.kernels.?[index] = kernel;
    kernels_set.programs.?[index] = program;

    return kernel;
}

inline fn get_index_with_complex_and_dtype(is_complex: bool, _: bool, dtype: wTensorDtype, _: anytype) usize {
    return @intFromBool(is_complex) * dtypes.number_of_dtypes + @as(usize, @intFromEnum(dtype));
}

pub fn get_cl_no_vector_kernel(
    command_queue: wCommandQueue, tensor: wTensor, kernel_id: wKernelsID, kernel_name: []const u8,
    kernel_source: []const u8, extra_args: ?[]const u8
) !cl.kernel.cl_kernel {
    const dtype = tensor.dtype;
    const is_complex = tensor.is_complex;

    return try create_and_get_kernel(
        command_queue, kernel_id, kernel_source, .{
            .dtype = dtype,
            .is_complex = is_complex,
            .vectors_enabled = false,
            .kernel_name = kernel_name,
            .extra_args = extra_args
        }, true, false, dtypes.number_of_dtypes * 2, get_index_with_complex_and_dtype,
        null
    );
}

inline fn get_index_with_dtype(_: bool, _: bool, dtype: wTensorDtype, _: anytype) usize {
    return @as(usize, @intFromEnum(dtype));
}

pub fn get_cl_no_vector_no_complex_single_kernel_per_dtype(
    command_queue: wCommandQueue, tensor: wTensor, kernel_id: wKernelsID, kernel_name: []const u8,
    kernel_source: []const u8, extra_args: ?[]const u8
) !cl.kernel.cl_kernel {
    const dtype = tensor.dtype;

    return try create_and_get_kernel(
        command_queue, kernel_id, kernel_source, .{
            .dtype = dtype,
            .is_complex = false,
            .vectors_enabled = false,
            .kernel_name = kernel_name,
            .extra_args = extra_args
        }, false, false, dtypes.number_of_dtypes, get_index_with_dtype,
        null
    );
}
