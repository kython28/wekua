const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;
const GemmAlgorithm = tensor_module.GemmAlgorithm;

const gemm_Kernel: []const u8 = @embedFile("kernels/generic_gemm.cl");
const gemm_nxn_Kernel: []const u8 = @embedFile("kernels/gemm_nxn.cl");
const gemm_nxn_gpu_Kernel: []const u8 = @embedFile("kernels/gemm_nxn_gpu.cl");


pub const Operation = enum(u8) {
    no_transpose = 0,
    transpose = 1,
    // ConjugateTranspose // TODO
};

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
    has_alpha: bool,
    has_beta: bool,
    op_a: Operation,
    op_b: Operation,
    default_algorithm: GemmAlgorithm,
    k_size: u64,
    algorithm_ptr: *GemmAlgorithm,
) TensorErrors!cl.kernel.Kernel {
    const SUPPORTED_TYPES = core.types.SUPPORTED_TYPES;
    const num_algorithms = std.meta.fields(GemmAlgorithm).len;
    const kernels_per_algorithm = 2 * 2 * 2 * 2 * 2 * SUPPORTED_TYPES.len;

    const kernels_set = try KernelsSet.getKernelSet(
        command_queue,
        .GEMM,
        num_algorithms * kernels_per_algorithm,
    );

    const algorithm: GemmAlgorithm = blk: {
        if (comptime core.types.isComplex(T)) {
            break :blk .generic;
        }
        if ((k_size % 32) == 0 and @intFromEnum(default_algorithm) >= @intFromEnum(GemmAlgorithm.@"32x32"))
            break :blk .@"32x32";
        if ((k_size % 16) == 0 and @intFromEnum(default_algorithm) >= @intFromEnum(GemmAlgorithm.@"16x16"))
            break :blk .@"16x16";
        if ((k_size % 8) == 0 and @intFromEnum(default_algorithm) >= @intFromEnum(GemmAlgorithm.@"8x8"))
            break :blk .@"8x8";
        if ((k_size % 4) == 0 and @intFromEnum(default_algorithm) >= @intFromEnum(GemmAlgorithm.@"4x4"))
            break :blk .@"4x4";
        break :blk .generic;
    };
    algorithm_ptr.* = algorithm;

    var kernel_index: usize = @intFromEnum(algorithm) * kernels_per_algorithm;
    kernel_index += @intFromBool(vectors_enabled) * (2 * 2 * 2 * 2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromBool(has_alpha) * (2 * 2 * 2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromBool(has_beta) * (2 * 2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromEnum(op_a) * (2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromEnum(op_b) * SUPPORTED_TYPES.len;
    kernel_index += @as(usize, core.types.getTypeIndex(T));

    if (kernels_set.kernels.?[kernel_index]) |v| return v;

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    const stride = @intFromEnum(algorithm) + 1;
    const block_size: u8 = switch (algorithm) {
        .generic => 2,
        .@"4x4" => 4,
        .@"8x8" => 8,
        .@"16x16" => 16,
        .@"32x32" => 32,
    };

    const allocator = command_queue.context.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(
        allocator,
        "-DHAS_ALPHA={d} -DHAS_BETA={d} -DA_TRANS={d} -DB_TRANS={d} -DSTRIDE={d} -DBLOCK_SIZE={d}",
        .{
            @intFromBool(has_alpha),
            @intFromBool(has_beta),
            @intFromEnum(op_a),
            @intFromEnum(op_b),
            stride,
            block_size,
        },
    );
    defer allocator.free(extra_args);

    const kernel_source: []const u8 = switch (algorithm) {
        .generic => gemm_Kernel,
        else => switch (command_queue.local_mem_type) {
            .local => gemm_nxn_gpu_Kernel,
            else => gemm_nxn_Kernel,
        },
    };

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .vectors_enabled = vectors_enabled,
            .kernel_name = "gemm",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        kernel_source,
    );

    kernels_set.kernels.?[kernel_index] = kernel;
    kernels_set.programs.?[kernel_index] = program;

    return kernel;
}

inline fn validateTensors(
    comptime T: type,
    a: *Tensor(T),
    b: *Tensor(T),
    c: *Tensor(T),
    op_a: Operation,
    op_b: Operation,
) TensorErrors!void {
    const a_shape = a.dimensions.shape;
    const b_shape = b.dimensions.shape;
    const c_shape = c.dimensions.shape;

    if (c_shape.len != 2 or a_shape.len != 2 or b_shape.len != 2) {
        return tensor_module.Errors.InvalidValue;
    }

    const a_m = a_shape[0];
    const a_k = a_shape[1];

    const b_k = b_shape[0];
    const b_n = b_shape[1];

    const c_m = c_shape[0];
    const c_n = c_shape[1];

    const match = switch (op_a) {
        .transpose => switch (op_b) {
            .no_transpose => (a_m == b_k and b_n == c_n and a_k == c_m),
            .transpose => (a_m == b_n and b_k == c_n and a_k == c_m),
        },
        .no_transpose => switch (op_b) {
            .no_transpose => (a_k == b_k and b_n == c_n and a_m == c_m),
            .transpose => (a_k == b_n and b_k == c_n and a_m == c_m),
        },
    };

    if (!match) {
        return tensor_module.Errors.InvalidValue;
    }
}

pub fn gemm(
    comptime T: type,
    pipeline: *Pipeline,
    alpha: ?T,
    a: *Tensor(T),
    op_a: Operation,
    b: *Tensor(T),
    op_b: Operation,
    beta: ?T,
    c: *Tensor(T),
) TensorErrors!void {
    try validateTensors(T, a, b, c, op_a, op_b);

    const command_queue = pipeline.command_queue;

    const has_alpha = (alpha != null or beta != null);
    const has_beta = (beta != null);

    const vectors_enabled = if (comptime core.types.isComplex(T))
        false
    else
        (a.flags.vectors_enabled and b.flags.vectors_enabled and op_a == .no_transpose and op_b == .transpose and command_queue.vector_widths[core.types.getTypeId(T)] > 1);


    var k_size: u64 = undefined;
    if (vectors_enabled) {
        k_size = a.memory_layout.row_pitch_for_vectors;
    }else{
        k_size = a.dimensions.shape[1 - @intFromEnum(op_a)];
        k_size += k_size % 2;
    }

    var algorithm: GemmAlgorithm = undefined;
    const kernel = try getKernel(
        T,
        command_queue,
        vectors_enabled,
        has_alpha,
        has_beta,
        op_a,
        op_b,
        c.work_configuration.gemm_algorithm_per_device[command_queue.wekua_id],
        k_size,
        &algorithm,
    );

    const prev_events = pipeline.prevEvents();

    var a_row_pitch: u64 = undefined;
    var b_row_pitch: u64 = undefined;

    if (vectors_enabled) {
        a_row_pitch = a.memory_layout.row_pitch_for_vectors;
        b_row_pitch = b.memory_layout.row_pitch_for_vectors;
    } else {
        a_row_pitch = a.memory_layout.row_pitch;
        b_row_pitch = b.memory_layout.row_pitch;
    }

    const wekua_id = command_queue.wekua_id;

    var global_work_items: []const u64 = &c.work_configuration.global_work_items_gemm_generic;
    var local_work_items: []const u64 = &c.work_configuration.local_work_items_gemm_generic[wekua_id];

    switch (algorithm) {
        .generic => {},
        .@"4x4" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_4x4[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_4x4[wekua_id];
        },
        .@"8x8" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_8x8[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_8x8[wekua_id];
        },
        .@"16x16" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_16x16[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_16x16[wekua_id];
        },
        .@"32x32" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_32x32[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_32x32[wekua_id];
        },
    }

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&a.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&b.buffer));
    try setArg(kernel, 2, cl_mem_size, @ptrCast(&c.buffer));

    try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&a_row_pitch));
    try setArg(kernel, 4, @sizeOf(u64), @ptrCast(&b_row_pitch));
    try setArg(kernel, 5, @sizeOf(u64), @ptrCast(&c.memory_layout.row_pitch));

    try setArg(kernel, 6, @sizeOf(u64), @ptrCast(&k_size));

    if (has_alpha) {
        const alpha_val: T = alpha orelse if (comptime core.types.isComplex(T))
            .{ .real = 1, .imag = 0 }
        else
            1;
        try setArg(kernel, 7, @sizeOf(T), @ptrCast(&alpha_val));

        if (has_beta) {
            const beta_val = beta.?;
            try setArg(kernel, 8, @sizeOf(T), @ptrCast(&beta_val));
        }
    }

    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        global_work_items,
        local_work_items,
        prev_events,
        &new_event,
    );
    errdefer tensor_module.helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

// -----------------------------------------------------------------------------
// Unit Tests
const testing = std.testing;

const memory = tensor_module.memory;
const fill = tensor_module.fill;
const identity_fn = tensor_module.identity;

fn castInt(comptime T: type, val: anytype) T {
    return switch (@typeInfo(T)) {
        .float => @floatFromInt(val),
        .int => @intCast(val),
        else => unreachable,
    };
}

fn castComplex(comptime T: type, real: anytype, imag: anytype) T {
    const Scalar = core.types.getType(T);
    return .{
        .real = castInt(Scalar, real),
        .imag = castInt(Scalar, imag),
    };
}

test "gemm - A * I = A for all non-complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                // Fill A with known values
                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // C = A * I
                try gemm(T, pipeline, null, a, .no_transpose, ident, .no_transpose, null, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                for (result, 0..) |val, i| {
                    try testing.expectEqual(castInt(T, i + 1), val);
                }
            }
        }
    }
}

test "gemm - I * A = A for all non-complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // C = I * A
                try gemm(T, pipeline, null, ident, .no_transpose, a, .no_transpose, null, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                for (result, 0..) |val, i| {
                    try testing.expectEqual(castInt(T, i + 1), val);
                }
            }
        }
    }
}

test "gemm - alpha scaling with identity" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // C = 2 * A * I
                try gemm(T, pipeline, 2, a, .no_transpose, ident, .no_transpose, null, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                for (result, 0..) |val, i| {
                    try testing.expectEqual(castInt(T, (i + 1) * 2), val);
                }
            }
        }
    }
}

test "gemm - alpha and beta" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // Pre-fill C with known values (all 1s)
                try fill.one(T, pipeline, c_mat);

                // C = 2 * A * I + 3 * C_old = 2*A + 3*1
                try gemm(T, pipeline, 2, a, .no_transpose, ident, .no_transpose, 3, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                for (result, 0..) |val, i| {
                    // 2*(i+1) + 3*1
                    try testing.expectEqual(castInt(T, (i + 1) * 2 + 3), val);
                }
            }
        }
    }
}

test "gemm - invalid shapes" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const config = tensor_module.CreateConfig{};

    // 2x3 * 4x5 → InvalidValue (inner dimensions don't match)
    {
        const a = try Tensor(f32).alloc(context, pipeline, &.{ 2, 3 }, config);
        defer a.release(pipeline);

        const b = try Tensor(f32).alloc(context, pipeline, &.{ 4, 5 }, config);
        defer b.release(pipeline);

        const c_mat = try Tensor(f32).alloc(context, pipeline, &.{ 2, 5 }, config);
        defer c_mat.release(pipeline);

        const err = gemm(f32, pipeline, null, a, .no_transpose, b, .no_transpose, null, c_mat);
        try testing.expectError(tensor_module.Errors.InvalidValue, err);
    }

    // 3D tensor → InvalidValue
    {
        const a = try Tensor(f32).alloc(context, pipeline, &.{ 2, 3, 4 }, config);
        defer a.release(pipeline);

        const b = try Tensor(f32).alloc(context, pipeline, &.{ 2, 3 }, config);
        defer b.release(pipeline);

        const c_mat = try Tensor(f32).alloc(context, pipeline, &.{ 2, 3 }, config);
        defer c_mat.release(pipeline);

        const err = gemm(f32, pipeline, null, a, .no_transpose, b, .no_transpose, null, c_mat);
        try testing.expectError(tensor_module.Errors.InvalidValue, err);
    }
}

test "gemm - transpose operations" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // C = A^T * I (transpose A)
                try gemm(T, pipeline, null, a, .transpose, ident, .no_transpose, null, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                // A^T * I = A^T, so result[i][j] = A[j][i] = j*n + i + 1
                for (0..n) |i| {
                    for (0..n) |j| {
                        const idx = i * n + j;
                        try testing.expectEqual(castInt(T, j * n + i + 1), result[idx]);
                    }
                }

                // C = I * A^T (transpose B which is identity, I^T = I, so C = A)
                try gemm(T, pipeline, null, ident, .no_transpose, a, .transpose, null, c_mat);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                // I * A^T = A^T, so result[i][j] = A[j][i] = j*n + i + 1
                for (0..n) |i| {
                    for (0..n) |j| {
                        const idx = i * n + j;
                        try testing.expectEqual(castInt(T, j * n + i + 1), result[idx]);
                    }
                }
            }
        }
    }
}

test "gemm - A * I = A for complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (comptime core.types.isComplex(T) and @typeInfo(core.types.getType(T)) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castComplex(T, i + 1, 0);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // C = A * I
                try gemm(T, pipeline, null, a, .no_transpose, ident, .no_transpose, null, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                for (result, 0..) |val, i| {
                    const expected = castComplex(T, i + 1, 0);
                    try testing.expectEqual(expected.real, val.real);
                    try testing.expectEqual(expected.imag, val.imag);
                }
            }
        }
    }
}

test "gemm - alpha scaling with identity for complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (comptime core.types.isComplex(T) and @typeInfo(core.types.getType(T)) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castComplex(T, i + 1, 0);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // C = {2, 0} * A * I
                const alpha_val: T = castComplex(T, 2, 0);
                try gemm(T, pipeline, alpha_val, a, .no_transpose, ident, .no_transpose, null, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                for (result, 0..) |val, i| {
                    const expected = castComplex(T, (i + 1) * 2, 0);
                    try testing.expectEqual(expected.real, val.real);
                    try testing.expectEqual(expected.imag, val.imag);
                }
            }
        }
    }
}

test "gemm - alpha and beta for complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 4;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (comptime core.types.isComplex(T) and @typeInfo(core.types.getType(T)) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const ident = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer ident.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castComplex(T, i + 1, 0);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try identity_fn(T, pipeline, ident);

                // Pre-fill C with ones
                try fill.one(T, pipeline, c_mat);

                // C = {2, 0} * A * I + {3, 0} * C_old = 2*A + 3*1
                const alpha_val: T = castComplex(T, 2, 0);
                const beta_val: T = castComplex(T, 3, 0);
                try gemm(T, pipeline, alpha_val, a, .no_transpose, ident, .no_transpose, beta_val, c_mat);

                const result = try allocator.alloc(T, n * n);
                defer allocator.free(result);

                try memory.writeToBuffer(T, pipeline, c_mat, result);
                pipeline.waitAndCleanup();

                for (result, 0..) |val, i| {
                    // 2*(i+1) + 3*1
                    const expected = castComplex(T, (i + 1) * 2 + 3, 0);
                    try testing.expectEqual(expected.real, val.real);
                    try testing.expectEqual(expected.imag, val.imag);
                }
            }
        }
    }
}
