const cl = @import("opencl");
const std = @import("std");

const core = @import("core");
const Context = core.Context;
const CommandQueue = core.CommandQueue;
const Pipeline = core.Pipeline;

const utils = @import("utils");

pub const helpers = @import("helpers.zig");

pub const fill = @import("fill.zig");
pub const memory = @import("memory/main.zig");
pub const random = @import("random/main.zig");
pub const transpose = @import("transpose.zig").transpose;
pub const convertions = @import("convertions/main.zig");
// pub const identity = @import("identity.zig").identity;
pub const print = @import("print.zig").print;

// const blas = @import("../blas/main.zig");

pub const Errors = error{
    InvalidValue,
    InvalidCoordinates,
    InvalidBuffer,
    UnqualTensorsAttribute,
    UnqualTensorsShape,
    UnqualTensorsDimension,
} || std.mem.Allocator.Error || cl.errors.OpenCLError || core.KernelsSet.Errors;

pub const CreateConfig = struct {
    cl_mem_flags: cl.buffer.MemFlags = cl.buffer.MemFlag.read_write,
    host_ptr: ?*anyopaque = null,
    vectors_enabled: bool = true,
};

const Dimensions = struct {
    shape: []u64,
    vl_shape: []u64,
    pitches: []u64,
    number_of_elements: u64,
    number_of_elements_without_padding: u64,
};

const WorkConfiguration = struct {
    global_work_items: [3]u64,
    global_work_items_without_vectors: [3]u64,

    local_work_items_1d: []u64,
    local_work_items_for_vectors_1d: []u64,

    local_work_items: [][3]u64,
    local_work_items_without_vectors: [][3]u64,

    // gemm_algorithm_per_device: []blas.gemm.Algorithm,
    global_work_items_gemm_generic: [2]u64,
    local_work_items_gemm_generic: [][2]u64,

    global_work_items_gemm_4x4: [][2]u64,
    local_work_items_gemm_4x4: [][2]u64,

    global_work_items_gemm_8x8: [][2]u64,
    local_work_items_gemm_8x8: [][2]u64,

    global_work_items_gemm_16x16: [][2]u64,
    local_work_items_gemm_16x16: [][2]u64,

    global_work_items_gemm_32x32: [][2]u64,
    local_work_items_gemm_32x32: [][2]u64,

    pub fn init(
        self: *WorkConfiguration,
        comptime T: type,
        arena_allocator: std.mem.Allocator,
        command_queues: []CommandQueue,
        depth: u64,
        penultimate_size: u64,
        padded_penultimate_size: u64,
        number_of_elements: u64,
        number_of_vectors: u64,
        last_size: u64,
        vl_shape: []const u64,
    ) error{OutOfMemory}!void {
        _ = T;
        const local_work_items_1d = try arena_allocator.alloc(u64, command_queues.len);
        const lobal_work_items_for_vectors_1d = try arena_allocator.alloc(u64, command_queues.len);
        const local_work_items = try arena_allocator.alloc([3]u64, command_queues.len);
        const local_work_items_without_vectors = try arena_allocator.alloc([3]u64, command_queues.len);

        self.local_work_items_1d = local_work_items_1d;
        self.local_work_items_for_vectors_1d = lobal_work_items_for_vectors_1d;
        self.local_work_items = local_work_items;
        self.local_work_items_without_vectors = local_work_items_without_vectors;

        const global_work_items: []u64 = &self.global_work_items;
        const global_work_items_without_vectors: []u64 = &self.global_work_items_without_vectors;

        global_work_items[0] = depth;
        global_work_items_without_vectors[0] = depth;

        global_work_items[1] = penultimate_size;
        global_work_items_without_vectors[1] = penultimate_size;

        global_work_items_without_vectors[2] = last_size;
        global_work_items[2] = vl_shape[vl_shape.len - 1];

        const local_work_items_gemm = try arena_allocator.alloc([2]u64, command_queues.len);
        self.local_work_items_gemm_generic = local_work_items_gemm;

        // const gemm_algorithm_per_device = try arena_allocator.alloc(blas.gemm.Algorithm, command_queues.len);
        // @memset(gemm_algorithm_per_device, blas.gemm.Algorithm.generic);
        // self.gemm_algorithm_per_device = gemm_algorithm_per_device;

        const gwi_h = padded_penultimate_size;
        const gwi_w = (last_size + (last_size % 2));

        self.global_work_items_gemm_generic[0] = gwi_h / 2;
        self.global_work_items_gemm_generic[1] = gwi_w / 2;

        for (
            command_queues,
            local_work_items_1d,
            lobal_work_items_for_vectors_1d,
            local_work_items,
            local_work_items_without_vectors,
            local_work_items_gemm,
        ) |cmd, *wa, *wv, *wmv, *wm, *lw_gemm| {
            utils.calculateWorkItems(
                @as([*]const u64, @ptrCast(&number_of_elements))[0..1],
                @as([*]u64, @ptrCast(wa))[0..1],
                cmd.max_work_group_size,
            );

            utils.calculateWorkItems(
                @as([*]const u64, @ptrCast(&number_of_vectors))[0..1],
                @as([*]u64, @ptrCast(wv))[0..1],
                cmd.max_work_group_size,
            );

            utils.calculateWorkItems(global_work_items, wmv, cmd.max_work_group_size);
            utils.calculateWorkItems(global_work_items_without_vectors, wm, cmd.max_work_group_size);
            utils.calculateWorkItems(
                &self.global_work_items_gemm_generic,
                lw_gemm,
                cmd.max_work_group_size,
            );
        }

        var max_block_length: u8 = 0;
        comptime var block_length = 4;
        inline while (block_length < 64) : (block_length *= 2) {
            if ((gwi_h % block_length == 0) and (gwi_w % block_length == 0)) {
                const algorithm_name = std.fmt.comptimePrint("{0}x{0}", .{block_length});
                const global_field_name = std.fmt.comptimePrint("global_work_items_gemm_{s}", .{algorithm_name});
                const local_field_name = std.fmt.comptimePrint("local_work_items_gemm_{s}", .{algorithm_name});

                @field(self, global_field_name) = try arena_allocator.alloc([2]u64, command_queues.len);
                @field(self, local_field_name) = try arena_allocator.alloc([2]u64, command_queues.len);
                max_block_length = block_length;
            }
        }

        // for (command_queues, 0..) |cmd, i| {
        //     const type_index = core.Context.getTypeId(T);
        //
        //     comptime var block_length2 = 4;
        //
        //     // var algorithm: blas.gemm.Algorithm = .generic;
        //     inline while (block_length2 < 64) : (block_length2 *= 2) {
        //         const block_size = cmd.vector_widths[type_index] * block_length2 * @sizeOf(T);
        //         const blocks_fit_in_local_mem = switch (cmd.local_mem_type) {
        //             .local => (block_size * 2 * 4 <= cmd.local_mem_size),
        //             .global => ((block_size * block_length2) <= 16 * 1024),
        //             else => unreachable,
        //         };
        //
        //         if (block_length2 <= max_block_length and blocks_fit_in_local_mem) {
        //             const algorithm_name = std.fmt.comptimePrint("{0}x{0}", .{block_length2});
        //             const g_field_name = std.fmt.comptimePrint("global_work_items_gemm_{s}", .{algorithm_name});
        //             const l_field_name = std.fmt.comptimePrint("local_work_items_gemm_{s}", .{algorithm_name});
        //
        //             const g_values = &@field(self, g_field_name)[i];
        //
        //             switch (cmd.local_mem_type) {
        //                 .local => @memcpy(g_values, &self.global_work_items_gemm_generic),
        //                 .global => {
        //                     g_values[0] = gwi_h / block_length2;
        //                     g_values[1] = gwi_w / block_length2;
        //                 },
        //                 else => unreachable,
        //             }
        //
        //             algorithm = @field(blas.gemm.Algorithm, algorithm_name);
        //
        //             utils.calculateWorkItems(g_values, &@field(self, l_field_name)[i], @min(block_length2 * block_length2, cmd.max_work_group_size));
        //         }
        //     }
        //
        //     self.gemm_algorithm_per_device[i] = algorithm;
        // }
    }
};

const MemoryLayout = struct {
    row_pitch: u64,
    row_pitch_for_vectors: u64,
    slice_pitch: u64,
    slice_pitch_for_vectors: u64,
    number_of_vectors: u64,
    size: usize,
};

const Flags = struct {
    vectors_enabled: bool,
};

pub fn Tensor(comptime T: type) type {
    const type_id = core.types.getTypeId(T);
    const is_complex = core.types.isComplex(T);

    return struct {
        context: *const Context,
        arena: std.heap.ArenaAllocator,

        buffer: cl.buffer.Mem,
        pitches_buffer: cl.buffer.Mem,

        dimensions: Dimensions,
        work_configuration: WorkConfiguration,
        memory_layout: MemoryLayout,
        flags: Flags,

        const Self = @This();

        fn createPitchBuffer(
            self: *Self,
            context: *const Context,
            pipeline: *Pipeline,
            pitches: []const u64,
        ) Errors!void {
            const pitches_buffer = try cl.buffer.create(
                context.cl_context,
                cl.buffer.MemFlag.read_write,
                pitches.len * @sizeOf(u64),
                null,
            );
            self.pitches_buffer = pitches_buffer;
            errdefer cl.buffer.release(pitches_buffer);

            const prev_events = pipeline.prevEvents();

            var new_event: cl.event.Event = undefined;
            try cl.buffer.write(
                pipeline.command_queue.cl_command_queue,
                pitches_buffer,
                false,
                0,
                pitches.len * @sizeOf(u64),
                pitches.ptr,
                prev_events,
                &new_event,
            );
            errdefer helpers.releaseEvent(new_event);

            try pipeline.append(&.{new_event});
        }

        pub fn empty(
            context: *const Context,
            pipeline: *Pipeline,
            shape: []const u64,
            config: CreateConfig,
        ) Errors!*Self {
            if (shape.len == 0) {
                return Errors.InvalidValue;
            }

            const allocator = context.allocator;
            const command_queues = context.command_queues;

            const tensor = try allocator.create(Self);
            errdefer allocator.destroy(tensor);

            tensor.context = context;
            tensor.arena = std.heap.ArenaAllocator.init(allocator);
            errdefer tensor.arena.deinit();

            const arena_allocator = tensor.arena.allocator();

            tensor.dimensions.shape = try arena_allocator.alloc(u64, shape.len);
            for (tensor.dimensions.shape, shape) |*d, s| {
                if (s == 0) return Errors.InvalidValue;

                d.* = s;
            }

            const vectors_enabled = (!is_complex and config.vectors_enabled);
            tensor.flags.vectors_enabled = vectors_enabled;

            var vector_width: u64 = 1;
            if (vectors_enabled) {
                for (command_queues) |cmd| {
                    const cw: u64 = @intCast(cmd.vector_widths[type_id]);
                    vector_width = @max(cw, vector_width);
                }
            }

            const vl_shape = try arena_allocator.dupe(u64, shape);
            tensor.dimensions.vl_shape = vl_shape;

            const ndim = shape.len;

            const last_element_index = ndim - 1;
            const penultimate_element_index = last_element_index -| 1;

            var number_of_elements_without_padding: u64 = 1;
            for (shape[0..penultimate_element_index]) |e| number_of_elements_without_padding *= e;

            const penultimate_size = if (ndim >= 2) shape[penultimate_element_index] else 1;
            const last_size = shape[last_element_index];

            const padded_penultimate_size = penultimate_size + (penultimate_size % 2);
            const depth: usize = number_of_elements_without_padding;

            number_of_elements_without_padding *= last_size * penultimate_size;
            tensor.dimensions.number_of_elements_without_padding = number_of_elements_without_padding;

            var row_pitch: u64 = last_size;
            if (!is_complex and vectors_enabled and vector_width > 1) {
                const remainder = @mod(row_pitch, vector_width);
                if (remainder > 0) {
                    row_pitch += vector_width - remainder;
                }
            }

            var row_pitch_for_vectors = row_pitch / vector_width;
            vl_shape[last_element_index] = row_pitch_for_vectors;

            const row_pitch_for_vectors_remainder = row_pitch_for_vectors % 2;
            row_pitch_for_vectors += row_pitch_for_vectors_remainder;
            row_pitch += vector_width * row_pitch_for_vectors_remainder;

            tensor.memory_layout.row_pitch = row_pitch;
            tensor.memory_layout.row_pitch_for_vectors = row_pitch_for_vectors;

            const slice_pitch = row_pitch * padded_penultimate_size;
            const number_of_elements = slice_pitch * depth;
            tensor.dimensions.number_of_elements = number_of_elements;
            tensor.memory_layout.slice_pitch = slice_pitch;
            tensor.memory_layout.slice_pitch_for_vectors = slice_pitch / vector_width;

            const number_of_vectors = number_of_elements / vector_width;
            tensor.memory_layout.number_of_vectors = number_of_vectors;

            const pitches = try arena_allocator.alloc(u64, shape.len);
            tensor.dimensions.pitches = pitches;

            const antepenultimate_element_index = penultimate_element_index -| 1;
            var pitch: u64 = number_of_elements;
            for (
                shape[0..antepenultimate_element_index],
                pitches[0..antepenultimate_element_index],
            ) |e, *p| {
                pitch /= e;
                p.* = pitch;
            }

            if (ndim >= 3) {
                pitches[antepenultimate_element_index] = slice_pitch;
            }

            if (ndim >= 2) {
                pitches[penultimate_element_index] = row_pitch;
            }

            pitches[last_element_index] = 1;

            try tensor.work_configuration.init(
                T,
                arena_allocator,
                command_queues,
                depth,
                penultimate_size,
                padded_penultimate_size,
                number_of_elements,
                number_of_vectors,
                last_size,
                vl_shape,
            );

            const size = number_of_elements * @sizeOf(T);
            tensor.memory_layout.size = size;

            tensor.buffer = try cl.buffer.create(
                context.cl_context,
                config.cl_mem_flags,
                size,
                config.host_ptr,
            );
            errdefer cl.buffer.release(tensor.buffer);

            try tensor.createPitchBuffer(context, pipeline, pitches);
            return tensor;
        }

        pub fn release(self: *Self, pipeline: *Pipeline) void {
            pipeline.waitAndCleanup();

            const allocator = self.context.allocator;

            cl.buffer.release(self.buffer);
            cl.buffer.release(self.pitches_buffer);

            self.arena.deinit();
            allocator.destroy(self);
        }

        pub fn alloc(
            context: *const Context,
            pipeline: *Pipeline,
            shape: []const u64,
            config: CreateConfig,
        ) Errors!*Self {
            const tensor = try empty(context, pipeline, shape, config);
            errdefer tensor.release(pipeline);

            try fill.zeroes(T, pipeline, tensor);

            return tensor;
        }
    };
}

// Unit Tests
const testing = std.testing;

test {
    std.testing.refAllDecls(@This());
}

test "Tensor.empty - basic initialization for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(tensor.context == context);
            try testing.expectEqual(@as(usize, 3), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 2), tensor.dimensions.shape[0]);
            try testing.expectEqual(@as(u64, 3), tensor.dimensions.shape[1]);
            try testing.expectEqual(@as(u64, 4), tensor.dimensions.shape[2]);
        }
    }
}

test "Tensor.empty - 1D tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 1), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 10), tensor.dimensions.shape[0]);

            try testing.expectEqual(tensor.dimensions.number_of_elements_without_padding, 10);
            try testing.expect(tensor.dimensions.number_of_elements >= 10);
        }
    }
}

test "Tensor.empty - 2D tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 5, 7 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 2), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 5), tensor.dimensions.shape[0]);
            try testing.expectEqual(@as(u64, 7), tensor.dimensions.shape[1]);
        }
    }
}

test "Tensor.empty - 4D tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4, 5 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 4), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 2), tensor.dimensions.shape[0]);
            try testing.expectEqual(@as(u64, 3), tensor.dimensions.shape[1]);
            try testing.expectEqual(@as(u64, 4), tensor.dimensions.shape[2]);
            try testing.expectEqual(@as(u64, 5), tensor.dimensions.shape[3]);
        }
    }
}

test "Tensor.empty - invalid empty shape" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const pipeline = try Pipeline.init(&context.command_queues[0]);
    defer pipeline.deinit();

    const shape: []const u64 = &.{};
    const config = CreateConfig{};

    const result = Tensor(f32).empty(context, pipeline, shape, config);
    try testing.expectError(Errors.InvalidValue, result);
}

test "Tensor.empty - invalid zero dimension" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const pipeline = try Pipeline.init(&context.command_queues[0]);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 0, 4 };
    const config = CreateConfig{};

    const result = Tensor(f32).empty(context, pipeline, &shape, config);
    try testing.expectError(Errors.InvalidValue, result);
}

test "Tensor.empty - complex flag disables vectors" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (core.types.isComplex(T) and command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(!tensor.flags.vectors_enabled);
        }
    }
}

test "Tensor.empty - vectors can be disabled" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{ .vectors_enabled = false };

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(!tensor.flags.vectors_enabled);
        }
    }
}

test "Tensor.alloc - creates zeroed tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(tensor.context == context);
            try testing.expectEqual(@as(usize, 3), tensor.dimensions.shape.len);
        }
    }
}

test "Tensor - dimensions calculated correctly" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Number of elements without padding should be 2*3*4 = 24
            try testing.expectEqual(@as(u64, 24), tensor.dimensions.number_of_elements_without_padding);

            // Total elements should be >= elements without padding (due to padding)
            try testing.expect(tensor.dimensions.number_of_elements >= tensor.dimensions.number_of_elements_without_padding);
        }
    }
}

test "Tensor - pitches calculated correctly" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Verify pitches array has same length as shape
            try testing.expectEqual(shape.len, tensor.dimensions.pitches.len);

            // Last pitch should equal multiplier (1 for non-complex)
            try testing.expectEqual(@as(u64, 1), tensor.dimensions.pitches[tensor.dimensions.pitches.len - 1]);
        }
    }
}

test "Tensor - memory layout" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Size should be at least the minimum required
            const min_size = tensor.dimensions.number_of_elements_without_padding * @sizeOf(T);
            try testing.expect(tensor.memory_layout.size >= min_size);

            // Row pitch should be >= last dimension
            try testing.expect(tensor.memory_layout.row_pitch >= shape[shape.len - 1]);
        }
    }
}

test "Tensor.release - proper cleanup" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            tensor.release(pipeline);
        }
    }
}

test "Tensor - multiple tensors on same context" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape1 = [_]u64{ 2, 3 };
    const shape2 = [_]u64{ 4, 5, 6 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor1 = try Tensor(T).alloc(context, pipeline, &shape1, config);
            defer tensor1.release(pipeline);

            const tensor2 = try Tensor(T).alloc(context, pipeline, &shape2, config);
            defer tensor2.release(pipeline);

            // Both should reference the same context
            try testing.expect(tensor1.context == tensor2.context);

            // But should have different shapes
            try testing.expect(tensor1.dimensions.shape.len != tensor2.dimensions.shape.len);
        }
    }
}

test "Tensor - work configuration initialized" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Verify work configuration arrays are allocated
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items.len);
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items_without_vectors.len);
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items_1d.len);
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items_for_vectors_1d.len);
        }
    }
}

test "Tensor.empty - custom memory flags" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{
        .cl_mem_flags = cl.buffer.MemFlag.read_only,
    };

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 2), tensor.dimensions.shape.len);
        }
    }
}

test "Tensor - vl_shape calculated" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // vl_shape should have same length as shape
            try testing.expectEqual(shape.len, tensor.dimensions.vl_shape.len);

            // First dimensions should match
            for (shape[0 .. shape.len - 1], tensor.dimensions.vl_shape[0 .. shape.len - 1]) |s, vl| {
                try testing.expectEqual(s, vl);
            }
        }
    }
}
