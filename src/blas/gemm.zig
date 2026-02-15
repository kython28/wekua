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
const gemm_nxn_outer_Kernel: []const u8 = @embedFile("kernels/gemm_nxn_outer.cl");
const gemm_nxn_gpu_Kernel: []const u8 = @embedFile("kernels/gemm_nxn_gpu.cl");
const gemm_pack_tiles_Kernel: []const u8 = @embedFile("kernels/gemm_pack.cl");
const gemm_nxn_pack_kernel: []const u8 = @embedFile("kernels/gemm_nxn_pack.cl");

pub const Operation = enum(u8) {
    no_transpose = 0,
    transpose = 1,
    // ConjugateTranspose // TODO
};

inline fn getBlockSizeFromAlgorithm(algorithm: GemmAlgorithm) u16 {
    return switch (algorithm) {
        .generic => 2,
        .@"4x4" => 4,
        .@"8x8" => 8,
        .@"16x16" => 16,
        .@"32x32" => 32,
        .@"64x64" => 64,
        .@"128x128" => 128,
        .@"256x256" => 256,
    };
}

inline fn getAlgorithm(
    comptime T: type,
    default_algorithm: GemmAlgorithm,
    k_size: u64,
) GemmAlgorithm {
    if (comptime core.types.isComplex(T)) {
        return .generic;
    }

    comptime var block_size = 256;
    inline while (block_size >= 2) {
        const field_name = switch (block_size) {
            2 => "generic",
            else => std.fmt.comptimePrint("{0}x{0}", .{block_size}),
        };
        const field_value = @field(GemmAlgorithm, field_name);
        if ((k_size % block_size) == 0 and @intFromEnum(default_algorithm) >= @intFromEnum(field_value)) {
            return field_value;
        }

        block_size /= 2;
    }

    @panic("Unsupported block size");
}

pub fn PackedTensors(comptime T: type) type {
    const TensorT = Tensor(T);

    const is_complex = core.types.isComplex(T);

    return struct {
        m_size: u64,
        n_size: u64,
        k_size: u64,

        packed_a: *TensorT,
        packed_b: *TensorT,
        vectors_enabled: bool,
        algorithm: GemmAlgorithm,

        const Self = @This();

        pub fn init(
            pipeline: *Pipeline,
            result_tensor: *Tensor(T),
            k_size: u64,
            vectors_enabled: bool,
        ) TensorErrors!*Self {
            const shape = result_tensor.dimensions.shape;
            if (shape.len != 2) {
                return tensor_module.Errors.InvalidValue;
            }

            const command_queue = pipeline.command_queue;
            const wekua_id = command_queue.wekua_id;

            const vector_width: u64 = @intCast(command_queue.vector_widths[core.types.getTypeId(T)]);
            var padded_k_size = k_size;
            if (!is_complex and vectors_enabled) {
                const remaining_size = padded_k_size % (vector_width * 2);
                if (remaining_size > 0) {
                    padded_k_size += (vector_width * 2) - remaining_size;
                }
                padded_k_size /= vector_width;
            } else {
                padded_k_size += padded_k_size % 2;
            }

            const recommended_algorithm = getAlgorithm(
                T,
                result_tensor.work_configuration.gemm_algorithm_per_device[wekua_id],
                padded_k_size,
            );
            const block_size = getBlockSizeFromAlgorithm(recommended_algorithm);

            const m_size = shape[0];
            const n_size = shape[1];

            const padded_m_size = m_size + (m_size % block_size);
            const padded_n_size = n_size + (n_size % block_size);

            const row_size = padded_k_size / block_size;
            var col_size: u64 = @as(u64, block_size) * block_size;
            if (!is_complex and vectors_enabled) {
                col_size *= vector_width;
            }

            const context = result_tensor.context;
            const packed_a = try TensorT.alloc(
                context,
                pipeline,
                &.{ padded_m_size / block_size, row_size, col_size },
                .{ .vectors_enabled = vectors_enabled },
            );
            errdefer packed_a.release(pipeline);

            const packed_b = try TensorT.alloc(
                context,
                pipeline,
                &.{ padded_n_size / block_size, row_size, col_size },
                .{ .vectors_enabled = vectors_enabled },
            );
            errdefer packed_b.release(pipeline);

            const self = try pipeline.allocator.create(Self);
            errdefer pipeline.allocator.destroy(self);

            self.* = .{
                .m_size = m_size,
                .n_size = n_size,
                .k_size = k_size,

                .packed_a = packed_a,
                .packed_b = packed_b,
                .vectors_enabled = !is_complex and vectors_enabled,
                .algorithm = recommended_algorithm,
            };

            return self;
        }

        fn getPackKernel(
            self: *const Self,
            command_queue: *const CommandQueue,
            transpose: bool,
        ) TensorErrors!cl.kernel.Kernel {
            const SUPPORTED_TYPES = core.types.SUPPORTED_TYPES;
            const num_algorithms = std.meta.fields(GemmAlgorithm).len;
            const kernels_per_algorithm = 2 * 2 * SUPPORTED_TYPES.len;

            const kernels_set = try KernelsSet.getKernelSet(
                command_queue,
                .PackGEMMTiles,
                num_algorithms * kernels_per_algorithm,
            );

            var kernel_index: usize = @intFromEnum(self.algorithm) * kernels_per_algorithm;
            kernel_index += @intFromBool(self.vectors_enabled) * (2 * SUPPORTED_TYPES.len);
            kernel_index += @intFromBool(transpose) * SUPPORTED_TYPES.len;
            kernel_index += @as(usize, core.types.getTypeIndex(T));

            if (kernels_set.kernels.?[kernel_index]) |v| return v;

            var kernel: cl.kernel.Kernel = undefined;
            var program: cl.program.Program = undefined;

            const block_size = getBlockSizeFromAlgorithm(self.algorithm);
            const allocator = command_queue.context.allocator;
            const extra_args = try std.fmt.allocPrint(
                allocator,
                "-DTRANSPOSE={d} -DBLOCK_SIZE={d}",
                .{ @intFromBool(transpose), block_size },
            );
            defer allocator.free(extra_args);

            try KernelsSet.compileKernel(
                T,
                command_queue,
                .{
                    .vectors_enabled = self.vectors_enabled,
                    .kernel_name = "pack",
                    .extra_args = extra_args,
                },
                &kernel,
                &program,
                gemm_pack_tiles_Kernel,
            );

            kernels_set.kernels.?[kernel_index] = kernel;
            kernels_set.programs.?[kernel_index] = program;

            return kernel;
        }

        inline fn validateTensors(
            self: *Self,
            a: *TensorT,
            op_a: Operation,
            b: *TensorT,
            op_b: Operation,
        ) TensorErrors!void {
            var valid = switch (op_a) {
                .no_transpose => (a.dimensions.shape[0] == self.n_size and a.dimensions.shape[1] == self.k_size),
                .transpose => (a.dimensions.shape[1] == self.n_size and a.dimensions.shape[0] == self.k_size),
            };

            valid &= switch (op_b) {
                .no_transpose => (b.dimensions.shape[0] == self.k_size and b.dimensions.shape[1] == self.m_size),
                .transpose => (b.dimensions.shape[1] == self.k_size and b.dimensions.shape[0] == self.m_size),
            };

            if (!valid) {
                return tensor_module.Errors.InvalidValue;
            }
        }

        pub fn pack(
            self: *Self,
            pipeline: *Pipeline,
            a: *TensorT,
            op_a: Operation,
            b: *TensorT,
            op_b: Operation,
        ) TensorErrors!void {
            try self.validateTensors(a, op_a, b, op_b);

            const command_queue = pipeline.command_queue;
            const wekua_id = command_queue.wekua_id;

            const a_transpose = (op_a == .transpose);
            const b_transpose = (op_b == .no_transpose); // inverted for B

            const kernel_a = try self.getPackKernel(command_queue, a_transpose);
            const kernel_b = try self.getPackKernel(command_queue, b_transpose);

            const setArg = cl.kernel.setArg;
            const cl_mem_size = @sizeOf(cl.buffer.Mem);

            const packed_a = self.packed_a;
            const packed_b = self.packed_b;

            var a_src_pitch: u64 = undefined;
            var a_dst_slice: u64 = undefined;
            var a_dst_pitch: u64 = undefined;

            var b_src_pitch: u64 = undefined;
            var b_dst_slice: u64 = undefined;
            var b_dst_pitch: u64 = undefined;

            if (self.vectors_enabled) {
                a_src_pitch = a.memory_layout.row_pitch_for_vectors;
                a_dst_slice = packed_a.memory_layout.slice_pitch_for_vectors;
                a_dst_pitch = packed_a.memory_layout.row_pitch_for_vectors;

                b_src_pitch = b.memory_layout.row_pitch_for_vectors;
                b_dst_slice = packed_b.memory_layout.slice_pitch_for_vectors;
                b_dst_pitch = packed_b.memory_layout.row_pitch_for_vectors;
            } else {
                a_src_pitch = a.memory_layout.row_pitch;
                a_dst_slice = packed_a.memory_layout.slice_pitch;
                a_dst_pitch = packed_a.memory_layout.row_pitch;

                b_src_pitch = b.memory_layout.row_pitch;
                b_dst_slice = packed_b.memory_layout.slice_pitch;
                b_dst_pitch = packed_b.memory_layout.row_pitch;
            }

            const a_global: []const u64 = self.packed_a.work_configuration.global_work_items[0..2];
            const a_local: []const u64 = self.packed_a.work_configuration.local_work_items[wekua_id][0..2];

            const b_global: []const u64 = self.packed_b.work_configuration.global_work_items[0..2];
            const b_local: []const u64 = self.packed_b.work_configuration.local_work_items[wekua_id][0..2];

            const prev_events = pipeline.prevEvents();

            try setArg(kernel_a, 0, cl_mem_size, @ptrCast(&a.buffer));
            try setArg(kernel_a, 1, cl_mem_size, @ptrCast(&self.packed_a.buffer));
            try setArg(kernel_a, 2, @sizeOf(u64), @ptrCast(&a_src_pitch));
            try setArg(kernel_a, 3, @sizeOf(u64), @ptrCast(&a_dst_slice));
            try setArg(kernel_a, 4, @sizeOf(u64), @ptrCast(&a_dst_pitch));

            var event_a: cl.event.Event = undefined;
            try cl.kernel.enqueueNdRange(
                command_queue.cl_command_queue,
                kernel_a,
                null,
                a_global,
                a_local,
                prev_events,
                &event_a,
            );
            errdefer tensor_module.helpers.releaseEvent(event_a);

            try setArg(kernel_b, 0, cl_mem_size, @ptrCast(&b.buffer));
            try setArg(kernel_b, 1, cl_mem_size, @ptrCast(&self.packed_b.buffer));
            try setArg(kernel_b, 2, @sizeOf(u64), @ptrCast(&b_src_pitch));
            try setArg(kernel_b, 3, @sizeOf(u64), @ptrCast(&b_dst_slice));
            try setArg(kernel_b, 4, @sizeOf(u64), @ptrCast(&b_dst_pitch));

            var event_b: cl.event.Event = undefined;
            try cl.kernel.enqueueNdRange(
                command_queue.cl_command_queue,
                kernel_b,
                null,
                b_global,
                b_local,
                prev_events,
                &event_b,
            );
            errdefer tensor_module.helpers.releaseEvent(event_b);

            try pipeline.append(&.{ event_a, event_b });
        }
    };
}

fn getGemmKernelWithoutPacking(
    comptime T: type,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
    has_alpha: bool,
    has_beta: bool,
    op_a: Operation,
    op_b: Operation,
    default_algorithm: GemmAlgorithm,
    a_row_pitch: u64,
    b_row_pitch: u64,
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

    const algorithm = getAlgorithm(T, default_algorithm, k_size);
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
    const block_size = getBlockSizeFromAlgorithm(algorithm);

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
            .global => blk: {
                if (op_a == .no_transpose and op_b == .transpose) {
                    break :blk gemm_nxn_Kernel;
                } else if (op_a == .transpose and op_b == .no_transpose) {
                    break :blk gemm_nxn_outer_Kernel;
                } else if (op_a == .transpose and op_b == .transpose) {
                    if (a_row_pitch > b_row_pitch) {
                        break :blk gemm_nxn_Kernel;
                    } else {
                        break :blk gemm_nxn_outer_Kernel;
                    }
                } else {
                    if (a_row_pitch > b_row_pitch) {
                        break :blk gemm_nxn_outer_Kernel;
                    } else {
                        break :blk gemm_nxn_Kernel;
                    }
                }
            },
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
    if (a.context != b.context or a.context != c.context) {
        return tensor_module.Errors.UnqualTensorsContext;
    }

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

fn gemmWithoutPacking(
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
    const command_queue = pipeline.command_queue;

    const has_alpha = (alpha != null or beta != null);
    const has_beta = (beta != null);

    var vectors_enabled: bool = a.flags.vectors_enabled and b.flags.vectors_enabled;
    const perfect_for_outer_product = (op_a == .transpose and op_b == .no_transpose and command_queue.local_mem_type == .global);
    if ((comptime core.types.isComplex(T)) or command_queue.vector_widths[core.types.getTypeId(T)] == 1) {
        vectors_enabled = false;
    } else {
        vectors_enabled &= ((op_a == .no_transpose and op_b == .transpose) or perfect_for_outer_product);
    }

    var k_size: u64 = undefined;
    if (vectors_enabled) {
        k_size = a.memory_layout.row_pitch_for_vectors;
    } else {
        k_size = a.dimensions.shape[1 - @intFromEnum(op_a)];
        k_size += k_size % 2;
    }

    if (perfect_for_outer_product and k_size < 4 and vectors_enabled) {
        vectors_enabled = false;

        k_size = a.dimensions.shape[1 - @intFromEnum(op_a)];
        k_size += k_size % 2;
    }

    var a_row_pitch: u64 = undefined;
    var b_row_pitch: u64 = undefined;

    if (vectors_enabled) {
        a_row_pitch = a.memory_layout.row_pitch_for_vectors;
        b_row_pitch = b.memory_layout.row_pitch_for_vectors;
    } else {
        a_row_pitch = a.memory_layout.row_pitch;
        b_row_pitch = b.memory_layout.row_pitch;
    }

    var algorithm: GemmAlgorithm = undefined;
    const kernel = try getGemmKernelWithoutPacking(
        T,
        command_queue,
        vectors_enabled,
        has_alpha,
        has_beta,
        op_a,
        op_b,
        c.work_configuration.gemm_algorithm_per_device[command_queue.wekua_id],
        a_row_pitch,
        b_row_pitch,
        k_size,
        &algorithm,
    );

    const prev_events = pipeline.prevEvents();
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
        .@"64x64" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_64x64[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_64x64[wekua_id];
        },
        .@"128x128" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_128x128[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_128x128[wekua_id];
        },
        .@"256x256" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_256x256[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_256x256[wekua_id];
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


fn getGemmKernelWithPacking(
    comptime T: type,
    command_queue: *const CommandQueue,
    vectors_enabled: bool,
    has_alpha: bool,
    has_beta: bool,
    op_a: Operation,
    op_b: Operation,
    algorithm: GemmAlgorithm,
) TensorErrors!cl.kernel.Kernel {
    const SUPPORTED_TYPES = core.types.SUPPORTED_TYPES;
    const num_algorithms = std.meta.fields(GemmAlgorithm).len;
    const kernels_per_algorithm = 2 * 2 * 2 * SUPPORTED_TYPES.len;

    const kernels_set = try KernelsSet.getKernelSet(
        command_queue,
        .GEMMPack,
        num_algorithms * kernels_per_algorithm,
    );

    var kernel_index: usize = @intFromEnum(algorithm) * kernels_per_algorithm;
    kernel_index += @intFromBool(vectors_enabled) * (2 * 2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromBool(has_alpha) * (2 * SUPPORTED_TYPES.len);
    kernel_index += @intFromBool(has_beta) * SUPPORTED_TYPES.len;
    kernel_index += @as(usize, core.types.getTypeIndex(T));

    if (kernels_set.kernels.?[kernel_index]) |v| return v;

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    const stride = @intFromEnum(algorithm) + 1;
    const block_size = getBlockSizeFromAlgorithm(algorithm);

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
        gemm_nxn_pack_kernel,
    );

    kernels_set.kernels.?[kernel_index] = kernel;
    kernels_set.programs.?[kernel_index] = program;

    return kernel;
}

fn gemmWithPacking(
    comptime T: type,
    pipeline: *Pipeline,
    alpha: ?T,
    a: *Tensor(T),
    op_a: Operation,
    b: *Tensor(T),
    op_b: Operation,
    beta: ?T,
    c: *Tensor(T),
    packed_tensors: *PackedTensors(T),
) TensorErrors!void {
    const command_queue = pipeline.command_queue;
    const wekua_id = command_queue.wekua_id;

    const has_alpha = (alpha != null or beta != null);
    const has_beta = (beta != null);

    try packed_tensors.pack(
        pipeline,
        a,
        op_a,
        b,
        op_b,
    );

    const vectors_enabled = packed_tensors.vectors_enabled;

    const algorithm = packed_tensors.algorithm;
    const kernel = try getGemmKernelWithPacking(
        T,
        command_queue,
        vectors_enabled,
        has_alpha,
        has_beta,
        op_a,
        op_b,
        algorithm,
    );

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
        .@"64x64" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_64x64[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_64x64[wekua_id];
        },
        .@"128x128" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_128x128[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_128x128[wekua_id];
        },
        .@"256x256" => {
            global_work_items = &c.work_configuration.global_work_items_gemm_256x256[wekua_id];
            local_work_items = &c.work_configuration.local_work_items_gemm_256x256[wekua_id];
        },
    }

    const packed_tensor_a = packed_tensors.packed_a;
    const packed_tensor_b = packed_tensors.packed_b;

    var A_slice_pitch: u64 = undefined;
    var A_row_pitch: u64 = undefined;

    var B_slice_pitch: u64 = undefined;
    var B_row_pitch: u64 = undefined;

    if (vectors_enabled) {
        A_slice_pitch = packed_tensor_a.memory_layout.slice_pitch_for_vectors;
        A_row_pitch = packed_tensor_a.memory_layout.row_pitch_for_vectors;

        B_slice_pitch = packed_tensor_b.memory_layout.slice_pitch_for_vectors;
        B_row_pitch = packed_tensor_b.memory_layout.row_pitch_for_vectors;
    } else {
        A_slice_pitch = packed_tensor_a.memory_layout.slice_pitch;
        A_row_pitch = packed_tensor_a.memory_layout.row_pitch;

        B_slice_pitch = packed_tensor_b.memory_layout.slice_pitch;
        B_row_pitch = packed_tensor_b.memory_layout.row_pitch;
    }

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&a.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&b.buffer));
    try setArg(kernel, 2, cl_mem_size, @ptrCast(&c.buffer));

    try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&A_slice_pitch));
    try setArg(kernel, 4, @sizeOf(u64), @ptrCast(&A_row_pitch));

    try setArg(kernel, 5, @sizeOf(u64), @ptrCast(&B_slice_pitch));
    try setArg(kernel, 6, @sizeOf(u64), @ptrCast(&B_row_pitch));

    try setArg(kernel, 7, @sizeOf(u64), @ptrCast(&c.memory_layout.row_pitch));

    try setArg(kernel, 6, @sizeOf(u64), @ptrCast(&packed_tensor_a.dimensions.shape[1]));

    if (has_alpha) {
        const alpha_val: T = alpha orelse if (comptime core.types.isComplex(T))
            .{ .real = 1, .imag = 0 }
        else
            1;
        try setArg(kernel, 8, @sizeOf(T), @ptrCast(&alpha_val));

        if (has_beta) {
            const beta_val = beta.?;
            try setArg(kernel, 9, @sizeOf(T), @ptrCast(&beta_val));
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
    packed_tensors: ?*PackedTensors(T),
) TensorErrors!void {
    try validateTensors(T, a, b, c, op_a, op_b);

    if (packed_tensors) |v| {
        try gemmWithPacking(
            T,
            pipeline,
            a,
            b,
            c,
            op_a,
            op_b,
            beta,
            c,
            v,
        );
    }else{
        try gemmWithoutPacking(
            T,
            pipeline,
            a,
            b,
            c,
            op_a,
            op_b,
            alpha,
            beta,
            c,
        );
    }
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

test "pack - normal packing (A no_transpose) for all non-complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try fill.zeroes(T, pipeline, b);

                var k_size: u64 = a.dimensions.shape[1];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .no_transpose, b, .no_transpose);

                const packed_a_elems = pack_ctx.packed_a.dimensions.number_of_elements_without_padding;
                const packed_a_buf = try allocator.alloc(T, packed_a_elems);
                defer allocator.free(packed_a_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_a, packed_a_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_a_shape = pack_ctx.packed_a.dimensions.shape;
                const tile_rows = packed_a_shape[0];
                const tile_cols = packed_a_shape[1];
                const tile_data = packed_a_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + y;
                                const src_col = tc * bs + x;
                                try testing.expectEqual(a_buf[src_row * n + src_col], packed_a_buf[packed_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "pack - transposed packing (A transpose) for all non-complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try fill.zeroes(T, pipeline, b);

                var k_size: u64 = a.dimensions.shape[0];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .transpose, b, .transpose);

                const packed_a_elems = pack_ctx.packed_a.dimensions.number_of_elements_without_padding;
                const packed_a_buf = try allocator.alloc(T, packed_a_elems);
                defer allocator.free(packed_a_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_a, packed_a_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_a_shape = pack_ctx.packed_a.dimensions.shape;
                const tile_rows = packed_a_shape[0];
                const tile_cols = packed_a_shape[1];
                const tile_data = packed_a_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + x;
                                const src_col = tc * bs + y;
                                try testing.expectEqual(a_buf[src_row * n + src_col], packed_a_buf[packed_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "pack - B packing with no_transpose for all non-complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const b_buf = try allocator.alloc(T, n * n);
                defer allocator.free(b_buf);

                for (b_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, b, b_buf);
                try fill.zeroes(T, pipeline, a);

                var k_size: u64 = a.dimensions.shape[1];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .no_transpose, b, .no_transpose);

                const packed_b_elems = pack_ctx.packed_b.dimensions.number_of_elements_without_padding;
                const packed_b_buf = try allocator.alloc(T, packed_b_elems);
                defer allocator.free(packed_b_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_b, packed_b_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_b_shape = pack_ctx.packed_b.dimensions.shape;
                const tile_rows = packed_b_shape[0];
                const tile_cols = packed_b_shape[1];
                const tile_data = packed_b_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + x;
                                const src_col = tc * bs + y;
                                try testing.expectEqual(b_buf[src_row * n + src_col], packed_b_buf[packed_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "pack - B packing with transpose for all non-complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and @typeInfo(T) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const b_buf = try allocator.alloc(T, n * n);
                defer allocator.free(b_buf);

                for (b_buf, 0..) |*val, i| {
                    val.* = castInt(T, i + 1);
                }

                try memory.readFromBuffer(T, pipeline, b, b_buf);
                try fill.zeroes(T, pipeline, a);

                var k_size: u64 = a.dimensions.shape[1];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .no_transpose, b, .transpose);

                const packed_b_elems = pack_ctx.packed_b.dimensions.number_of_elements_without_padding;
                const packed_b_buf = try allocator.alloc(T, packed_b_elems);
                defer allocator.free(packed_b_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_b, packed_b_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_b_shape = pack_ctx.packed_b.dimensions.shape;
                const tile_rows = packed_b_shape[0];
                const tile_cols = packed_b_shape[1];
                const tile_data = packed_b_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + y;
                                const src_col = tc * bs + x;
                                try testing.expectEqual(b_buf[src_row * n + src_col], packed_b_buf[packed_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "pack - normal packing (A no_transpose) for complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (comptime core.types.isComplex(T) and @typeInfo(core.types.getType(T)) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castComplex(T, i + 1, 0);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try fill.zeroes(T, pipeline, b);

                var k_size: u64 = a.dimensions.shape[1];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .no_transpose, b, .no_transpose);

                const packed_a_elems = pack_ctx.packed_a.dimensions.number_of_elements_without_padding;
                const packed_a_buf = try allocator.alloc(T, packed_a_elems);
                defer allocator.free(packed_a_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_a, packed_a_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_a_shape = pack_ctx.packed_a.dimensions.shape;
                const tile_rows = packed_a_shape[0];
                const tile_cols = packed_a_shape[1];
                const tile_data = packed_a_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + y;
                                const src_col = tc * bs + x;
                                const expected = a_buf[src_row * n + src_col];
                                const actual = packed_a_buf[packed_idx];
                                try testing.expectEqual(expected.real, actual.real);
                                try testing.expectEqual(expected.imag, actual.imag);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "pack - transposed packing (A transpose) for complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (comptime core.types.isComplex(T) and @typeInfo(core.types.getType(T)) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const a_buf = try allocator.alloc(T, n * n);
                defer allocator.free(a_buf);

                for (a_buf, 0..) |*val, i| {
                    val.* = castComplex(T, i + 1, 0);
                }

                try memory.readFromBuffer(T, pipeline, a, a_buf);
                try fill.zeroes(T, pipeline, b);

                var k_size: u64 = a.dimensions.shape[0];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .transpose, b, .transpose);

                const packed_a_elems = pack_ctx.packed_a.dimensions.number_of_elements_without_padding;
                const packed_a_buf = try allocator.alloc(T, packed_a_elems);
                defer allocator.free(packed_a_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_a, packed_a_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_a_shape = pack_ctx.packed_a.dimensions.shape;
                const tile_rows = packed_a_shape[0];
                const tile_cols = packed_a_shape[1];
                const tile_data = packed_a_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + x;
                                const src_col = tc * bs + y;
                                const expected = a_buf[src_row * n + src_col];
                                const actual = packed_a_buf[packed_idx];
                                try testing.expectEqual(expected.real, actual.real);
                                try testing.expectEqual(expected.imag, actual.imag);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "pack - B packing with no_transpose for complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (comptime core.types.isComplex(T) and @typeInfo(core.types.getType(T)) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const b_buf = try allocator.alloc(T, n * n);
                defer allocator.free(b_buf);

                for (b_buf, 0..) |*val, i| {
                    val.* = castComplex(T, i + 1, 0);
                }

                try memory.readFromBuffer(T, pipeline, b, b_buf);
                try fill.zeroes(T, pipeline, a);

                var k_size: u64 = a.dimensions.shape[1];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .no_transpose, b, .no_transpose);

                const packed_b_elems = pack_ctx.packed_b.dimensions.number_of_elements_without_padding;
                const packed_b_buf = try allocator.alloc(T, packed_b_elems);
                defer allocator.free(packed_b_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_b, packed_b_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_b_shape = pack_ctx.packed_b.dimensions.shape;
                const tile_rows = packed_b_shape[0];
                const tile_cols = packed_b_shape[1];
                const tile_data = packed_b_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + x;
                                const src_col = tc * bs + y;
                                const expected = b_buf[src_row * n + src_col];
                                const actual = packed_b_buf[packed_idx];
                                try testing.expectEqual(expected.real, actual.real);
                                try testing.expectEqual(expected.imag, actual.imag);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "pack - B packing with transpose for complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const n = 8;
    const shape = [_]u64{ n, n };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (comptime core.types.isComplex(T) and @typeInfo(core.types.getType(T)) == .float) {
            if (command_queue.isTypeSupported(T)) {
                const a = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer a.release(pipeline);

                const b = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer b.release(pipeline);

                const c_mat = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer c_mat.release(pipeline);

                const b_buf = try allocator.alloc(T, n * n);
                defer allocator.free(b_buf);

                for (b_buf, 0..) |*val, i| {
                    val.* = castComplex(T, i + 1, 0);
                }

                try memory.readFromBuffer(T, pipeline, b, b_buf);
                try fill.zeroes(T, pipeline, a);

                var k_size: u64 = a.dimensions.shape[1];
                k_size += k_size % 2;

                const pack_ctx = try PackedTensors(T).init(pipeline, c_mat, k_size, true);
                defer pack_ctx.packed_a.release(pipeline);
                defer pack_ctx.packed_b.release(pipeline);
                defer pipeline.allocator.destroy(pack_ctx);

                try pack_ctx.pack(pipeline, a, .no_transpose, b, .transpose);

                const packed_b_elems = pack_ctx.packed_b.dimensions.number_of_elements_without_padding;
                const packed_b_buf = try allocator.alloc(T, packed_b_elems);
                defer allocator.free(packed_b_buf);

                try memory.writeToBuffer(T, pipeline, pack_ctx.packed_b, packed_b_buf);
                pipeline.waitAndCleanup();

                const bs: u64 = getBlockSizeFromAlgorithm(pack_ctx.algorithm);
                const packed_b_shape = pack_ctx.packed_b.dimensions.shape;
                const tile_rows = packed_b_shape[0];
                const tile_cols = packed_b_shape[1];
                const tile_data = packed_b_shape[2];

                for (0..tile_rows) |tr| {
                    for (0..tile_cols) |tc| {
                        for (0..bs) |y| {
                            for (0..bs) |x| {
                                const packed_idx = tr * tile_cols * tile_data + tc * tile_data + y * bs + x;
                                const src_row = tr * bs + y;
                                const src_col = tc * bs + x;
                                const expected = b_buf[src_row * n + src_col];
                                const actual = packed_b_buf[packed_idx];
                                try testing.expectEqual(expected.real, actual.real);
                                try testing.expectEqual(expected.imag, actual.imag);
                            }
                        }
                    }
                }
            }
        }
    }
}
