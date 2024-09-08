const std = @import("std");
const cl = @import("opencl");

const wCommandQueue = @import("command_queue.zig").wCommandQueue;
const wTensorDtype = @import("../tensor/utils/dtypes.zig").wTensorDtype;

const header_content: []const u8 = @embedFile("wekua_cl_lib.h");

pub const wKernelsID = enum (u16) {
    Random = 0,
    RandRange = 1,
    Transpose = 2
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
