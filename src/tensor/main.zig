const cl = @import("opencl");
const std = @import("std");

const core = @import("../core/main.zig");
const Context = core.Context;
const CommandQueue = core.CommandQueue;

pub const EventManager = @import("event_manager.zig");
const utils = @import("../utils/utils.zig");

pub const helpers = @import("helpers.zig");

pub const fill = @import("fill.zig");
pub const memory = @import("memory/main.zig");
pub const random = @import("random/main.zig");
pub usingnamespace @import("transpose.zig");
pub const convertions = @import("convertions/main.zig");
pub usingnamespace @import("identity.zig");

const blas = @import("../blas/main.zig");

pub const Errors = error{
    InvalidValue,
    InvalidCoordinates,
    TensorDoesNotSupportComplexNumbers,
    InvalidBuffer,
    UnqualTensorsAttribute,
    UnqualTensorsShape,
    UnqualTensorsDimension,
};

pub const CreateConfig = struct {
    cl_mem_flags: cl.buffer.cl_mem_flags = @intFromEnum(cl.buffer.enums.mem_flags.read_write),
    host_ptr: ?*anyopaque = null,
    is_complex: bool = false,
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

    gemm_algorithm: blas.gemm.Algorithm,
    global_work_items_gemm_generic: [2]u64,
    local_work_items_gemm_generic: [][2]u64,

    global_work_items_gemm_4x4: [2]u64,
    local_work_items_gemm_4x4: [][2]u64,

    global_work_items_gemm_8x8: [2]u64,
    local_work_items_gemm_8x8: [][2]u64,

    global_work_items_gemm_16x16: [2]u64,
    local_work_items_gemm_16x16: [][2]u64,

    global_work_items_gemm_32x32: [2]u64,
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
        vector_width: u64,
    ) !void {
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

        self.gemm_algorithm = .generic;

        const gwi_h = padded_penultimate_size / 2;
        const gwi_w = (last_size + (last_size % 2)) / 2;

        self.global_work_items_gemm_generic[0] = gwi_h;
        self.global_work_items_gemm_generic[1] = gwi_w;

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

        if ((gwi_h % 2 == 0) and (gwi_w % 2 == 0) and (vector_width * 4 * 4 * @sizeOf(T)) < 16 * 1024) {
            self.global_work_items_gemm_4x4[0] = gwi_h / 2;
            self.global_work_items_gemm_4x4[1] = gwi_w / 2;

            self.gemm_algorithm = .@"4x4";

            self.local_work_items_gemm_4x4 = try arena_allocator.alloc([2]u64, command_queues.len);
            for (command_queues, self.local_work_items_gemm_4x4) |cmd, *lw_gemm| {
                _ = cmd;
                utils.calculateWorkItems(
                    &self.global_work_items_gemm_4x4,
                    lw_gemm,
                    16,
                );
            }
        }

        if ((gwi_h % 4 == 0) and (gwi_w % 4 == 0) and (vector_width * 8 * 8 * @sizeOf(T)) < 16 * 1024) {
            self.global_work_items_gemm_8x8[0] = gwi_h / 4;
            self.global_work_items_gemm_8x8[1] = gwi_w / 4;

            self.gemm_algorithm = .@"8x8";

            self.local_work_items_gemm_8x8 = try arena_allocator.alloc([2]u64, command_queues.len);
            for (command_queues, self.local_work_items_gemm_8x8) |cmd, *lw_gemm| {
                _ = cmd;
                utils.calculateWorkItems(
                    &self.global_work_items_gemm_8x8,
                    lw_gemm,
                    64,
                );
            }
        }

        if ((gwi_h % 8 == 0) and (gwi_w % 8 == 0) and (vector_width * 16 * 16 * @sizeOf(T)) < 16 * 1024) {
            self.global_work_items_gemm_16x16[0] = gwi_h / 8;
            self.global_work_items_gemm_16x16[1] = gwi_w / 8;

            self.gemm_algorithm = .@"16x16";

            self.local_work_items_gemm_16x16 = try arena_allocator.alloc([2]u64, command_queues.len);
            for (command_queues, self.local_work_items_gemm_16x16) |cmd, *lw_gemm| {
                _ = cmd;
                utils.calculateWorkItems(
                    &self.global_work_items_gemm_16x16,
                    lw_gemm,
                    128,
                );
            }
        }

        if ((gwi_h % 16 == 0) and (gwi_w % 16 == 0) and (vector_width * 32 * 32 * @sizeOf(T)) < 16 * 1024) {
            self.global_work_items_gemm_32x32[0] = gwi_h / 16;
            self.global_work_items_gemm_32x32[1] = gwi_w / 16;

            self.gemm_algorithm = .@"32x32";

            self.local_work_items_gemm_32x32 = try arena_allocator.alloc([2]u64, command_queues.len);
            for (command_queues, self.local_work_items_gemm_32x32) |cmd, *lw_gemm| {
                _ = cmd;
                utils.calculateWorkItems(
                    &self.global_work_items_gemm_32x32,
                    lw_gemm,
                    256,
                );
            }
        }
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
    is_complex: bool,
    vectors_enabled: bool,
};

pub fn Tensor(comptime T: type) type {
    switch (T) {
        i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 => {},
        else => @compileError("Type not supported"),
    }

    const type_index = core.Context.getTypeId(T);

    return struct {
        context: *const Context,
        arena: std.heap.ArenaAllocator,

        buffer: cl.buffer.cl_mem,
        pitches_buffer: cl.buffer.cl_mem,

        dimensions: Dimensions,
        work_configuration: WorkConfiguration,
        memory_layout: MemoryLayout,
        flags: Flags,

        events_manager: EventManager,

        const this = @This();

        fn createPitchBuffer(self: *this, context: *const Context, pitches: []const u64) !void {
            const pitches_buffer = try cl.buffer.create(
                context.ctx,
                @intFromEnum(cl.buffer.enums.mem_flags.read_write),
                pitches.len * @sizeOf(u64),
                null,
            );
            self.pitches_buffer = pitches_buffer;
            errdefer cl.buffer.release(pitches_buffer);

            const prev_events = self.events_manager.getPrevEvents(.write);

            var new_event: cl.event.cl_event = undefined;
            const command_queue = context.command_queues[0];
            try cl.buffer.write(
                command_queue.cmd,
                pitches_buffer,
                false,
                0,
                pitches.len * @sizeOf(u64),
                pitches.ptr,
                prev_events,
                &new_event,
            );
            errdefer |err| helpers.releaseEvent(new_event, err);

            _ = try self.events_manager.appendNewEvent(.write, prev_events, new_event, null);
        }

        pub fn empty(context: *Context, shape: []const u64, config: CreateConfig) !*this {
            if (shape.len == 0) {
                return Errors.InvalidValue;
            }

            const allocator = context.allocator;
            const command_queues = context.command_queues;

            const tensor = try allocator.create(this);
            errdefer allocator.destroy(tensor);

            tensor.context = context;
            try tensor.events_manager.init(allocator, &context.events_batch_queue);
            errdefer tensor.events_manager.deinit();

            tensor.arena = std.heap.ArenaAllocator.init(allocator);
            errdefer tensor.arena.deinit();

            const arena_allocator = tensor.arena.allocator();

            tensor.dimensions.shape = try arena_allocator.alloc(u64, shape.len);
            for (tensor.dimensions.shape, shape) |*d, s| {
                if (s == 0) return Errors.InvalidValue;

                d.* = s;
            }

            const is_complex = config.is_complex;
            const vectors_enabled = if (is_complex) false else config.vectors_enabled;

            tensor.flags.is_complex = is_complex;
            tensor.flags.vectors_enabled = vectors_enabled;

            const multiplier: usize = if (is_complex) 2 else 1;

            var vector_width: u64 = 1;
            if (vectors_enabled) {
                for (command_queues) |cmd| {
                    const cw: u64 = @intCast(cmd.vector_widths[type_index]);
                    vector_width = @max(cw, vector_width);
                }
            }

            const vl_shape = try arena_allocator.dupe(u64, shape);
            tensor.dimensions.vl_shape = vl_shape;

            const ndim = shape.len;

            const last_element_index = ndim -| 1;
            const penultimate_element_index = last_element_index -| 1;

            var number_of_elements_without_padding: u64 = 1;
            for (shape[0..penultimate_element_index]) |e| number_of_elements_without_padding *= e;

            const penultimate_size = if (ndim >= 2) shape[penultimate_element_index] else 1;
            const last_size = shape[last_element_index];

            const padded_penultimate_size = penultimate_size + (penultimate_size % 2);
            const depth: usize = number_of_elements_without_padding;

            number_of_elements_without_padding *= multiplier * last_size * penultimate_size;
            tensor.dimensions.number_of_elements_without_padding = number_of_elements_without_padding;

            var row_pitch: u64 = last_size;
            if (vectors_enabled and vector_width > 1) {
                const remainder = @mod(row_pitch, vector_width);
                if (remainder > 0) {
                    row_pitch += vector_width - remainder;
                }
            }

            var row_pitch_for_vectors = row_pitch / vector_width;
            vl_shape[last_element_index] = row_pitch_for_vectors;
            row_pitch_for_vectors *= multiplier;
            row_pitch *= multiplier;

            const row_pitch_for_vectors_remainder = row_pitch_for_vectors % (multiplier * 2);
            if (row_pitch_for_vectors_remainder > 0) {
                const diff = multiplier * 2 - row_pitch_for_vectors_remainder;
                row_pitch_for_vectors += diff;
                row_pitch += vector_width * diff;
            }

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

            pitches[last_element_index] = multiplier;

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
                vector_width,
            );

            const size = number_of_elements * @sizeOf(T);
            tensor.memory_layout.size = size;

            tensor.buffer = try cl.buffer.create(context.ctx, config.cl_mem_flags, size, config.host_ptr);
            errdefer cl.buffer.release(tensor.buffer);

            try tensor.createPitchBuffer(context, pitches);
            return tensor;
        }

        pub fn release(self: *this) void {
            const allocator = self.context.allocator;

            self.events_manager.deinit();

            cl.buffer.release(self.buffer);
            cl.buffer.release(self.pitches_buffer);

            self.arena.deinit();
            allocator.destroy(self);
        }

        pub fn alloc(context: *Context, shape: []const u64, config: CreateConfig) !*this {
            const tensor = try empty(context, shape, config);
            errdefer tensor.release();

            const prev_events = tensor.events_manager.getPrevEvents(.write);

            const zero: T = 0;
            const command_queue = context.command_queues[0];
            const cmd = command_queue.cmd;

            var new_event: cl.event.cl_event = undefined;
            try cl.buffer.fill(
                cmd,
                tensor.buffer,
                &zero,
                @sizeOf(T),
                0,
                tensor.memory_layout.size,
                prev_events,
                &new_event,
            );
            errdefer |err| helpers.releaseEvent(new_event, err);

            _ = try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
            return tensor;
        }

        pub fn wait(self: *this) !void {
            const prev_events = self.events_manager.getPrevEvents(.write) orelse return;
            try cl.event.wait_for_many(prev_events);
        }
    };
}
