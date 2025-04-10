const builtin = @import("builtin");
const std = @import("std");
const cl = @import("opencl");

const core = @import("main.zig");
const w_tensor = @import("../tensor/main.zig");

const CommandQueue = core.CommandQueue;
const Tensor = w_tensor.Tensor;

const header_content: []const u8 = @embedFile("wekua_cl_lib.cl");

pub const KernelsID = enum(u16) {
    Fill,
    RandomUniform,
    RandRange,
    Transpose,
    ToComplex,
    ToReal,
    AXPY,
    Identity,
    GEMM,

    // --- Math kernels ---
    // Basic
    Dot,
    Sum,

    // Trigonometric
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,

    // --- Activation kernels ---
    Sigmoid,
    SigmoidDev,
    TanhDev,

    // --- Layer kernels ---
    LinearBias,
    LinearBiasStep,

    // --- Loss kernels ---
    MSE,

    // --- Optimizer kernels ---
    GDM,
    Adagrad,
    RMSProp,
};

pub const total_number_of_kernels = @typeInfo(KernelsID).@"enum".fields.len;
pub const total_number_of_headers = core.SupportedTypes.len * 2 * 2;

kernels: []?cl.kernel.cl_kernel = &.{},
programs: []?cl.program.cl_program = &.{},
initialized: bool = false,

pub const CompileOptions = struct {
    is_complex: bool = false,
    vectors_enabled: bool = true,
    kernel_name: []const u8,
    extra_args: ?[]const u8 = null,
};

fn compileHeader(
    comptime T: type,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
    is_complex: bool,
) !cl.program.cl_program {
    const type_index: u16 = core.getTypeId(T);
    const index: u16 = type_index * 4 + @as(u16, @intFromBool(vectors_enabled)) * 2 + @intFromBool(is_complex);

    const headers = &command_queue.headers;
    const programs = headers.programs;
    if (programs[index]) |prg| {
        return prg;
    }

    const new_header_prg = try cl.program.create_with_source(
        command_queue.ctx.ctx,
        @as([*]const []const u8, @ptrCast(&header_content))[0..1],
        command_queue.allocator,
    );

    programs[index] = new_header_prg;
    return new_header_prg;
}

fn showBuildLog(program: cl.program.cl_program, command_queue: *const CommandQueue) !void {
    var msg_len: usize = undefined;
    try cl.program.get_build_info(
        program,
        command_queue.device,
        .build_log,
        0,
        null,
        &msg_len,
    );

    const compile_log: []u8 = try command_queue.allocator.alloc(u8, msg_len);
    defer command_queue.allocator.free(compile_log);

    try cl.program.get_build_info(
        program,
        command_queue.device,
        .build_log,
        msg_len,
        compile_log.ptr,
        null,
    );

    if (builtin.is_test) {
        std.log.warn("An error while compiling the kernel:\n{s}", .{compile_log});
    } else {
        std.log.err("An error while compiling the kernel:\n{s}", .{compile_log});
    }
}

pub fn compileKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    options: CompileOptions,
    kernel: *cl.kernel.cl_kernel,
    program: *cl.program.cl_program,
    content: []const u8,
) !void {
    const cl_ctx = command_queue.ctx.ctx;
    const allocator = command_queue.allocator;
    const new_program = try cl.program.create_with_source(
        cl_ctx,
        @as([*]const []const u8, @ptrCast(&content))[0..1],
        allocator,
    );
    defer cl.program.release(new_program);

    const type_index = core.getTypeId(T);
    const vector_width = blk: {
        if (options.vectors_enabled and !options.is_complex) {
            break :blk command_queue.vector_widths[type_index];
        }
        break :blk 1;
    };

    if (vector_width == 0) return error.TypeNotSupported;

    comptime var default_arguments: []const u8 = "-DWK_VECTOR_WIDTH={d} -DWK_DTYPE={d} -DWK_MEM_TYPE={d} -DWK_COMPLEX={d} -DWK_CACHE_LINE_SIZE={d}";
    switch (builtin.mode) {
        .Debug => default_arguments = default_arguments ++ " -cl-opt-disable -g",
        .ReleaseFast, .ReleaseSmall => default_arguments = default_arguments ++ " -cl-fast-relaxed-math",
        else => {},
    }

    var args: []u8 = undefined;
    if (options.extra_args) |v| {
        args = try std.fmt.allocPrint(
            allocator,
            default_arguments ++ " {s}\x00",
            .{
                vector_width,
                type_index,
                @intFromEnum(command_queue.local_mem_type),
                @intFromBool(options.is_complex),
                command_queue.cache_line_size,
                v,
            },
        );
    } else {
        args = try std.fmt.allocPrint(
            allocator,
            default_arguments ++ "\x00",
            .{
                vector_width,
                type_index,
                @intFromEnum(command_queue.local_mem_type),
                @intFromBool(options.is_complex),
                command_queue.cache_line_size,
            },
        );
    }
    defer allocator.free(args);

    const header_prg = try compileHeader(
        T,
        command_queue,
        options.vectors_enabled,
        options.is_complex,
    );
    const header_name: []const u8 = "wekua.h";

    const devices: []const cl.device.cl_device_id = @as(
        [*]const cl.device.cl_device_id,
        @ptrCast(&command_queue.device),
    )[0..1];

    cl.program.compile(
        allocator,
        new_program,
        devices,
        args,
        @as([*]const cl.program.cl_program, @ptrCast(&header_prg))[0..1],
        @as([*]const []const u8, @ptrCast(&header_name))[0..1],
        null,
        null,
    ) catch |err| {
        switch (err) {
            error.compile_program_failure => {
                showBuildLog(new_program, command_queue) catch |err2| {
                    std.log.err("Unexpected error while showing build log: {s}", .{@errorName(err2)});
                    std.log.warn("No able to show build log", .{});
                };
            },
            else => {},
        }
        return err;
    };

    program.* = cl.program.link(
        cl_ctx,
        devices,
        null,
        @as([*]const cl.program.cl_program, @ptrCast(&new_program))[0..1],
        null,
        null,
    ) catch |err| {
        switch (err) {
            error.link_program_failure => {
                showBuildLog(new_program, command_queue) catch |err2| {
                    std.log.err("Unexpected error while showing build log: {s}", .{@errorName(err2)});
                    std.log.warn("No able to show build log", .{});
                };
            },
            else => {},
        }
        return err;
    };

    kernel.* = try cl.kernel.create(program.*, options.kernel_name);
}

pub fn getKernelSet(
    command_queue: *const CommandQueue,
    kernel_id: KernelsID,
    comptime number_of_cl_kernels: usize,
) !*const KernelSet {
    const kernels_set = &command_queue.kernels[@intFromEnum(kernel_id)];
    if (kernels_set.initialized) {
        return kernels_set;
    }

    const allocator = command_queue.allocator;

    const cl_kernels = try allocator.alloc(?cl.kernel.cl_kernel, number_of_cl_kernels);
    errdefer allocator.free(cl_kernels);

    @memset(cl_kernels, null);

    const cl_programs = try allocator.alloc(?cl.program.cl_program, number_of_cl_kernels);
    errdefer allocator.free(cl_programs);

    @memset(cl_programs, null);

    const mutable_cl_kernels_set: *KernelSet = @constCast(kernels_set);
    mutable_cl_kernels_set.kernels = cl_kernels;
    mutable_cl_kernels_set.programs = cl_programs;
    mutable_cl_kernels_set.initialized = true;

    return kernels_set;
}

pub fn createAndGetKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    kernel_id: KernelsID,
    kernel_source: []const u8,
    options: CompileOptions,
    comptime can_use_complex: bool,
    comptime can_use_vectors: bool,
    comptime number_of_cl_kernels: usize,
    kernel_index: usize,
) !cl.kernel.cl_kernel {
    if (!can_use_complex and options.is_complex) {
        @panic("Kernels with complex numbers are not allowed");
    }
    if (!can_use_vectors and options.vectors_enabled) {
        @panic("Kernels with vectors are not allowed");
    }

    const kernels_set = try getKernelSet(command_queue, kernel_id, number_of_cl_kernels);
    if (kernels_set.kernels[kernel_index]) |kernel| {
        return kernel;
    }

    var kernel: cl.kernel.cl_kernel = undefined;
    var program: cl.program.cl_program = undefined;

    try compileKernel(T, command_queue, options, &kernel, &program, kernel_source);

    kernels_set.kernels[kernel_index] = kernel;
    kernels_set.programs[kernel_index] = program;

    return kernel;
}

pub fn getClKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *const Tensor(T),
    kernel_id: KernelsID,
    kernel_name: []const u8,
    kernel_source: []const u8,
    extra_args: ?[]const u8,
) !cl.kernel.cl_kernel {
    const is_complex = tensor.flags.is_complex;
    const vectors_enabled = tensor.flags.vectors_enabled;

    const kernel_index = (@intFromBool(vectors_enabled) * (2 * core.SupportedTypes.len) +
        @intFromBool(is_complex) * core.SupportedTypes.len + @as(usize, core.getTypeId(T)));

    return try createAndGetKernel(
        T,
        command_queue,
        kernel_id,
        kernel_source,
        CompileOptions{
            .is_complex = is_complex,
            .vectors_enabled = vectors_enabled,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        true,
        true,
        core.SupportedTypes.len * 2 * 2,
        kernel_index,
    );
}

pub fn getClNoVectorKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *const Tensor(T),
    kernel_id: KernelsID,
    kernel_name: []const u8,
    kernel_source: []const u8,
    extra_args: ?[]const u8,
) !cl.kernel.cl_kernel {
    const is_complex = tensor.flags.is_complex;

    const type_index: usize = core.getTypeId(T);
    const kernel_index = @intFromBool(is_complex) * core.SupportedTypes.len + type_index;

    return try createAndGetKernel(
        T,
        command_queue,
        kernel_id,
        kernel_source,
        .{
            .is_complex = is_complex,
            .vectors_enabled = false,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        true,
        false,
        core.SupportedTypes.len * 2,
        kernel_index,
    );
}

pub fn getClNoVectorNoComplexSingleKernelPerDtype(
    comptime T: type,
    command_queue: *const CommandQueue,
    tensor: *const Tensor(T),
    kernel_id: KernelsID,
    kernel_name: []const u8,
    kernel_source: []const u8,
    extra_args: ?[]const u8,
) !cl.kernel.cl_kernel {
    const dtype = tensor.dtype;

    return try createAndGetKernel(
        T,
        command_queue,
        kernel_id,
        kernel_source,
        .{
            .dtype = dtype,
            .is_complex = false,
            .vectors_enabled = false,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        false,
        false,
        w_tensor.SupportedTypes.len,
        w_tensor.getTypeId(T),
    );
}

const KernelSet = @This();
