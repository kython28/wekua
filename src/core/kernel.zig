const builtin = @import("builtin");
const std = @import("std");
const cl = @import("opencl");

const core = @import("main.zig");
const CommandQueue = core.CommandQueue;

pub const wekua_header: []const u8 = @embedFile("wekua_cl_lib.cl");
pub const Errors = error{TypeNotSupported} || cl.errors.OpenCLError || std.mem.Allocator.Error;

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

pub const TOTAL_NUMBER_OF_KERNELS = @typeInfo(KernelsID).@"enum".fields.len;
pub const TOTAL_NUMBER_OF_HEADERS = core.types.SupportedTypes.len * 2;

kernels: ?[]?cl.kernel.Kernel = null,
programs: ?[]?cl.program.Program = null,
initialized: bool = false,

pub const CompileOptions = struct {
    vectors_enabled: bool = true,
    kernel_name: []const u8,
    extra_args: ?[]const u8 = null,
};

fn compileHeader(
    comptime T: type,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
) Errors!cl.program.Program {
    const type_index: u16 = core.types.getTypeIndex(T);
    const index: u16 = type_index * 2 + @as(u16, @intFromBool(vectors_enabled));

    const headers = &command_queue.headers;
    const programs = headers.programs;
    if (programs.?[index]) |prg| {
        return prg;
    }

    const new_header_prg = try cl.program.createWithSource(
        command_queue.context.cl_context,
        @as([*]const []const u8, @ptrCast(&wekua_header))[0..1],
        command_queue.context.allocator,
    );

    programs.?[index] = new_header_prg;
    return new_header_prg;
}

fn showBuildLog(program: cl.program.Program, command_queue: *const CommandQueue) Errors!void {
    var msg_len: usize = undefined;
    try cl.program.get_build_info(
        program,
        command_queue.device,
        .build_log,
        0,
        null,
        &msg_len,
    );

    const allocator = command_queue.context.allocator;
    const compile_log: []u8 = try allocator.alloc(u8, msg_len);
    defer allocator.free(compile_log);

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

pub fn compileCustomKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    options: CompileOptions,
    program: *cl.program.Program,
    headers: []const cl.program.Program,
    header_names: []const []const u8,
    source_codes: []const []const u8,
) Errors!cl.kernel.Kernel {
    const context = command_queue.context;
    const cl_ctx = context.cl_context;
    const allocator = context.allocator;
    const new_program = try cl.program.createWithSource(
        cl_ctx,
        source_codes,
        allocator,
    );
    defer cl.program.release(new_program);

    const is_complex = comptime core.types.isComplex(T);
    const type_index = core.types.getTypeIndex(T);
    const vector_width = blk: {
        if (!is_complex and options.vectors_enabled) {
            break :blk command_queue.vector_widths[type_index];
        }
        break :blk 1;
    };

    if (vector_width == 0) return error.TypeNotSupported;


    comptime var default_arguments: []const u8 = std.fmt.comptimePrint("-DWK_VECTOR_WIDTH={{d}} -DWK_DTYPE={d} -DWK_DTYPE_ID={d} -DWK_MEM_TYPE={{d}} -DWK_COMPLEX={d} -DWK_CACHE_LINE_SIZE={{d}}", .{
        type_index,
        core.types.getTypeId(T),
        @intFromBool(is_complex)
    });
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
                @intFromEnum(command_queue.local_mem_type),
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
                @intFromEnum(command_queue.local_mem_type),
                command_queue.cache_line_size,
            },
        );
    }
    defer allocator.free(args);

    const devices: []const cl.device.DeviceId = @as(
        [*]const cl.device.DeviceId,
        @ptrCast(&command_queue.device),
    )[0..1];

    cl.program.compile(
        allocator,
        new_program,
        devices,
        args,
        headers,
        header_names,
        null,
        null,
    ) catch |err| {
        switch (err) {
            error.CompileProgramFailure => {
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
        &.{new_program},
        null,
        null,
    ) catch |err| {
        switch (err) {
            error.LinkProgramFailure => {
                showBuildLog(new_program, command_queue) catch |err2| {
                    std.log.err(
                        "Unexpected error while showing build log: {s}",
                        .{@errorName(err2)},
                    );
                    std.log.warn("No able to show build log", .{});
                };
            },
            else => {},
        }
        return err;
    };

    const kernel = try cl.kernel.create(program.*, options.kernel_name);
    return kernel;
}

pub fn compileKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    options: CompileOptions,
    kernel: *cl.kernel.Kernel,
    program: *cl.program.Program,
    source_code: []const u8,
) Errors!void {
    const header_prg = try compileHeader(
        T,
        command_queue,
        options.vectors_enabled,
    );
    const header_name: []const u8 = "wekua.h";

    kernel.* = try compileCustomKernel(
        T,
        command_queue,
        options,
        program,
        &.{header_prg},
        &.{header_name},
        &.{source_code},
    );
}

pub fn getKernelSet(
    command_queue: *const CommandQueue,
    kernel_id: KernelsID,
    number_of_cl_kernels: usize,
) Errors!*const KernelSet {
    const kernels_set = &command_queue.kernels[@intFromEnum(kernel_id)];
    if (kernels_set.initialized) {
        return kernels_set;
    }

    const allocator = command_queue.context.allocator;

    const cl_kernels = try allocator.alloc(?cl.kernel.Kernel, number_of_cl_kernels);
    errdefer allocator.free(cl_kernels);

    @memset(cl_kernels, null);

    const cl_programs = try allocator.alloc(?cl.program.Program, number_of_cl_kernels);
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
    number_of_cl_kernels: usize,
    kernel_index: usize,
) Errors!cl.kernel.Kernel {
    const is_complex = comptime core.types.isComplex(T);
    if (!can_use_complex and is_complex) {
        @compileError("Kernels with complex numbers are not allowed");
    }

    if (!can_use_vectors and options.vectors_enabled) {
        @panic("Kernels with vectors are not allowed");
    }

    const kernels_set = try getKernelSet(command_queue, kernel_id, number_of_cl_kernels);
    if (kernels_set.kernels.?[kernel_index]) |kernel| {
        return kernel;
    }

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    try compileKernel(T, command_queue, options, &kernel, &program, kernel_source);

    kernels_set.kernels.?[kernel_index] = kernel;
    kernels_set.programs.?[kernel_index] = program;

    return kernel;
}

pub fn getClKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
    kernel_id: KernelsID,
    kernel_name: []const u8,
    kernel_source: []const u8,
    extra_args: ?[]const u8,
) Errors!cl.kernel.Kernel {
    const kernel_index = (@intFromBool(vectors_enabled) * core.types.SupportedTypes.len +
        @as(usize, core.types.getTypeIndex(T)));

    return createAndGetKernel(
        T,
        command_queue,
        kernel_id,
        kernel_source,
        CompileOptions{
            .vectors_enabled = vectors_enabled,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        true,
        true,
        core.types.SupportedTypes.len * 2,
        kernel_index,
    );
}

pub fn getClNoVectorKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    kernel_id: KernelsID,
    kernel_name: []const u8,
    kernel_source: []const u8,
    extra_args: ?[]const u8,
) Errors!cl.kernel.Kernel {
    const type_index: usize = core.types.getTypeIndex(T);

    return createAndGetKernel(
        T,
        command_queue,
        kernel_id,
        kernel_source,
        .{
            .vectors_enabled = false,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        true,
        false,
        core.types.SupportedTypes.len,
        type_index,
    );
}

pub fn getClNoVectorNoComplexSingleKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    kernel_id: KernelsID,
    kernel_name: []const u8,
    kernel_source: []const u8,
    extra_args: ?[]const u8,
) Errors!cl.kernel.Kernel {
    return createAndGetKernel(
        T,
        command_queue,
        kernel_id,
        kernel_source,
        .{
            .vectors_enabled = false,
            .kernel_name = kernel_name,
            .extra_args = extra_args,
        },
        false,
        false,
        core.types.SupportedTypes.len / 2,
        core.types.getTypeIndex(T),
    );
}

const KernelSet = @This();

// Unit Tests
const testing = std.testing;

test "compileHeader - basic functionality" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    // Test compiling header for f32 type
    const header_program = try compileHeader(f32, command_queue, true);

    // Test that subsequent calls return the same program (cached)
    const header_program2 = try compileHeader(f32, command_queue, true);
    try testing.expectEqual(header_program, header_program2);
}

test "compileHeader - different type combinations" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    // Test different combinations should produce different programs
    const header1 = try compileHeader(f32, command_queue, true);
    const header2 = try compileHeader(core.types.ComplexF32, command_queue, true); // complex
    const header3 = try compileHeader(f32, command_queue, false); // no vectors
    const header4 = try compileHeader(i32, command_queue, true); // different type

    // All should be different programs
    try testing.expect(header1 != header2);
    try testing.expect(header1 != header3);
    try testing.expect(header1 != header4);
    try testing.expect(header2 != header3);
    try testing.expect(header2 != header4);
    try testing.expect(header3 != header4);
}

test "getKernelSet - initialization and caching" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    // Test getting a kernel set
    const kernel_set = try getKernelSet(command_queue, KernelsID.Fill, 10);
    try testing.expect(kernel_set.initialized);
    try testing.expect(kernel_set.kernels != null);
    try testing.expect(kernel_set.programs != null);
    try testing.expectEqual(@as(usize, 10), kernel_set.kernels.?.len);
    try testing.expectEqual(@as(usize, 10), kernel_set.programs.?.len);

    // Test that subsequent calls return the same kernel set (cached)
    const kernel_set2 = try getKernelSet(command_queue, KernelsID.Fill, 10);
    try testing.expectEqual(kernel_set, kernel_set2);
}

test "getKernelSet - different kernel IDs produce different sets" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const kernel_set1 = try getKernelSet(command_queue, KernelsID.Fill, 5);
    const kernel_set2 = try getKernelSet(command_queue, KernelsID.AXPY, 5);

    try testing.expect(kernel_set1 != kernel_set2);
}

test "compileKernel - basic compilation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    // Simple test kernel source
    const test_kernel_source =
        \\__kernel void test_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = gid;
        \\}
    ;

    const options = CompileOptions{
        .vectors_enabled = false,
        .kernel_name = "test_kernel",
    };

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    try compileKernel(f32, command_queue, options, &kernel, &program, test_kernel_source);

    // Cleanup
    cl.kernel.release(kernel);
    cl.program.release(program);
}

test "compileCustomKernel - with headers" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    // Get a header program
    const header_program = try compileHeader(f32, command_queue, false);

    const test_kernel_source =
        \\#include "wekua.h"
        \\__kernel void test_custom_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = (float)gid;
        \\}
    ;

    const options = CompileOptions{
        .vectors_enabled = false,
        .kernel_name = "test_custom_kernel",
    };

    var program: cl.program.Program = undefined;

    const kernel = try compileCustomKernel(
        f32,
        command_queue,
        options,
        &program,
        &.{header_program},
        &.{"wekua.h"},
        &.{test_kernel_source},
    );

    // Cleanup
    cl.kernel.release(kernel);
    cl.program.release(program);
}

test "createAndGetKernel - caching behavior" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const test_kernel_source =
        \\__kernel void cache_test_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = gid * 2.0f;
        \\}
    ;

    const options = CompileOptions{
        .vectors_enabled = false,
        .kernel_name = "cache_test_kernel",
    };

    // First call should compile the kernel
    const kernel1 = try createAndGetKernel(
        f32,
        command_queue,
        KernelsID.Fill,
        test_kernel_source,
        options,
        false, // can_use_complex
        false, // can_use_vectors
        core.types.SupportedTypes.len,
        core.types.getTypeIndex(f32),
    );

    // Second call should return the cached kernel
    const kernel2 = try createAndGetKernel(
        f32,
        command_queue,
        KernelsID.Fill,
        test_kernel_source,
        options,
        false,
        false,
        core.types.SupportedTypes.len,
        core.types.getTypeIndex(f32),
    );

    try testing.expectEqual(kernel1, kernel2);
}

test "getClKernel - index calculation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    // Skip if f32 is not supported
    if (!command_queue.isTypeSupported(f32)) return;

    const test_kernel_source =
        \\__kernel void index_test_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = gid;
        \\}
    ;

    _ = try getClKernel(
        f32,
        command_queue,
        true, // vectors_enabled
        KernelsID.Fill,
        "index_test_kernel",
        test_kernel_source,
        null,
    );
}

test "getClNoVectorKernel - no vector support" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const test_kernel_source =
        \\__kernel void no_vector_test_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = gid + 1.0f;
        \\}
    ;

    _ = try getClNoVectorKernel(
        f32,
        command_queue,
        KernelsID.AXPY,
        "no_vector_test_kernel",
        test_kernel_source,
        null,
    );
}

test "getClNoVectorNoComplexSingleKernel - simplest case" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const test_kernel_source =
        \\__kernel void simple_test_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = 42.0f;
        \\}
    ;

    _ = try getClNoVectorNoComplexSingleKernel(
        f32,
        command_queue,
        KernelsID.Sum,
        "simple_test_kernel",
        test_kernel_source,
        null,
    );
}

test "CompileOptions with extra_args" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const test_kernel_source =
        \\#ifdef CUSTOM_DEFINE
        \\__kernel void extra_args_test_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = CUSTOM_VALUE;
        \\}
        \\#else
        \\__kernel void extra_args_test_kernel(__global float* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = 0.0f;
        \\}
        \\#endif
    ;

    _ = try getClNoVectorNoComplexSingleKernel(
        f32,
        command_queue,
        KernelsID.Identity,
        "extra_args_test_kernel",
        test_kernel_source,
        "-DCUSTOM_DEFINE -DCUSTOM_VALUE=123.0f",
    );
}

test "Multiple kernel types compilation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];

    const test_kernel_source =
        \\#include "wekua.h"
        \\__kernel void multi_type_test_kernel(__global wks* data) {
        \\    int gid = get_global_id(0);
        \\    data[gid] = (wks)gid;
        \\}
    ;

    // Test compilation for different supported types
    const TestTypes = [_]type{ f32, i32, u32 };

    inline for (TestTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            _ = try getClNoVectorNoComplexSingleKernel(
                T,
                command_queue,
                KernelsID.Dot,
                "multi_type_test_kernel",
                test_kernel_source,
                null,
            );
        }
    }
}
