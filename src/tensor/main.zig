const cl = @import("opencl");
const std = @import("std");

const core = @import("../core/main.zig");
const Context = core.Context;

pub const EventManager = @import("event_manager.zig");
const utils = @import("../utils/utils.zig");

const helpers = @import("helpers.zig");

pub const fill = @import("fill.zig");
pub const memory = @import("memory/main.zig");
pub const random = @import("random/main.zig");
pub usingnamespace @import("transpose.zig");
// const convertions = @import("convertions/main.zig");

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

pub fn Tensor(comptime T: type) type {
    switch (T) {
        i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 => {},
        else => @compileError("Type not supported"),
    }

    const type_index = core.Context.getTypeId(T);

    return struct {
        context: *const Context,

        buffer: cl.buffer.cl_mem,

        shape: []u64,
        vl_shape: []u64,

        number_of_elements: u64,
        number_of_elements_without_padding: u64,
        number_of_vectors: u64,

        row_pitch: u64,
        pitches: []u64,
        pitches_buffer: cl.buffer.cl_mem,

        row_pitch_for_vectors: u64,

        size: usize,

        is_complex: bool,
        vectors_enabled: bool,

        shape_like_matrix: [2]u64,
        shape_like_matrix_without_vectors: [2]u64,

        work_item_for_all_elements: []u64,
        work_item_for_all_vectors: []u64,
        work_items_for_matrix_shape: [][2]u64,
        work_items_for_matrix_shape_without_vectors: [][2]u64,

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

            try self.events_manager.appendNewEvent(.write, prev_events, new_event, null);
        }

        pub fn empty(context: *const Context, shape: []const u64, config: CreateConfig) !*this {
            const allocator = context.allocator;
            const command_queues = context.command_queues;

            const tensor = try allocator.create(this);
            errdefer allocator.destroy(tensor);

            tensor.context = context;
            try tensor.events_manager.init(allocator);
            errdefer tensor.events_manager.deinit();

            tensor.shape = try allocator.alloc(u64, shape.len);
            errdefer allocator.free(tensor.shape);
            for (tensor.shape, shape) |*d, s| {
                if (s == 0) return Errors.InvalidValue;

                d.* = s;
            }

            const is_complex = config.is_complex;
            const vectors_enabled = if (is_complex) false else config.vectors_enabled;

            tensor.is_complex = is_complex;
            tensor.vectors_enabled = vectors_enabled;

            const multiplier: usize = if (is_complex) 2 else 1;

            var vector_width: u64 = 1;
            if (vectors_enabled) {
                for (command_queues) |cmd| {
                    const cw: u64 = @intCast(cmd.vector_widths[type_index]);
                    vector_width = @max(cw, vector_width);
                }
            }

            const vl_shape = try allocator.dupe(u64, shape);
            errdefer allocator.free(vl_shape);
            tensor.vl_shape = vl_shape;

            const last_element_index = shape.len - 1;
            var row_pitch: u64 = shape[last_element_index] * multiplier;
            if (vectors_enabled and vector_width > 1) {
                row_pitch += vector_width - @mod(row_pitch, vector_width);
            }

            const row_pitch_for_vectors = row_pitch / vector_width;
            tensor.row_pitch = row_pitch;
            tensor.row_pitch_for_vectors = row_pitch_for_vectors;

            @memcpy(vl_shape[0..last_element_index], shape[0..last_element_index]);
            vl_shape[last_element_index] = row_pitch_for_vectors;

            var number_of_elements: u64 = 1;
            for (shape[0..last_element_index]) |e| number_of_elements *= e;

            const number_of_elements_without_padding = number_of_elements * shape[last_element_index] * multiplier;
            tensor.number_of_elements_without_padding = number_of_elements_without_padding;

            number_of_elements *= row_pitch;
            tensor.number_of_elements = number_of_elements;

            const number_of_vectors = number_of_elements / vector_width;
            tensor.number_of_vectors = number_of_vectors;

            const pitches = try allocator.alloc(u64, shape.len);
            errdefer allocator.free(pitches);
            tensor.pitches = pitches;

            var pitch: u64 = number_of_elements;
            for (shape[0..last_element_index], pitches[0..last_element_index]) |e, *p| {
                pitch /= e;
                p.* = pitch;
            }
            pitches[last_element_index] = multiplier;

            const work_item_for_all_elements = try allocator.alloc(u64, command_queues.len);
            errdefer allocator.free(work_item_for_all_elements);

            const work_item_for_all_vectors = try allocator.alloc(u64, command_queues.len);
            errdefer allocator.free(work_item_for_all_vectors);

            const work_items_for_matrix_shape = try allocator.alloc([2]u64, command_queues.len);
            errdefer allocator.free(work_items_for_matrix_shape);

            const work_items_for_matrix_shape_without_vectors = try allocator.alloc([2]u64, command_queues.len);
            errdefer allocator.free(work_items_for_matrix_shape_without_vectors);

            tensor.work_item_for_all_elements = work_item_for_all_elements;
            tensor.work_item_for_all_vectors = work_item_for_all_vectors;
            tensor.work_items_for_matrix_shape = work_items_for_matrix_shape;
            tensor.work_items_for_matrix_shape_without_vectors = work_items_for_matrix_shape_without_vectors;

            const rows = number_of_elements / row_pitch;
            const shape_like_matrix: []u64 = &tensor.shape_like_matrix;
            shape_like_matrix[0] = rows;
            shape_like_matrix[1] = vl_shape[last_element_index];

            const shape_like_matrix_without_vectors: []u64 = &tensor.shape_like_matrix_without_vectors;
            shape_like_matrix_without_vectors[0] = rows;
            shape_like_matrix_without_vectors[1] = shape[last_element_index];

            for (
                command_queues,
                work_item_for_all_elements,
                work_item_for_all_vectors,
                work_items_for_matrix_shape,
                work_items_for_matrix_shape_without_vectors,
            ) |cmd, *wa, *wv, *wmv, *wm| {
                utils.calculate_work_items(
                    @as([*]const u64, @ptrCast(&number_of_elements))[0..1],
                    @as([*]u64, @ptrCast(wa))[0..1],
                    cmd.max_work_group_size,
                );

                utils.calculate_work_items(
                    @as([*]const u64, @ptrCast(&number_of_vectors))[0..1],
                    @as([*]u64, @ptrCast(wv))[0..1],
                    cmd.max_work_group_size,
                );

                utils.calculate_work_items(shape_like_matrix, wmv, cmd.max_work_group_size);
                utils.calculate_work_items(shape_like_matrix_without_vectors, wm, cmd.max_work_group_size);
            }

            const size = number_of_elements * @sizeOf(T);
            tensor.size = size;

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

            allocator.free(self.shape);
            allocator.free(self.vl_shape);
            allocator.free(self.pitches);
            allocator.free(self.work_item_for_all_elements);
            allocator.free(self.work_item_for_all_vectors);
            allocator.free(self.work_items_for_matrix_shape);
            allocator.free(self.work_items_for_matrix_shape_without_vectors);
            allocator.destroy(self);
        }

        pub fn alloc(context: *const Context, shape: []const u64, config: CreateConfig) !*this {
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
                tensor.size,
                prev_events,
                &new_event,
            );
            errdefer {
                cl.event.wait(new_event) catch |err| {
                    std.debug.panic("Unexpected error while waiting for event: {s}", .{@errorName(err)});
                };
                cl.event.release(new_event);
            }

            try tensor.events_manager.appendNewEvent(.write, prev_events, new_event, null);
            return tensor;
        }

        pub fn wait(self: *this) !void {
            const prev_events = self.events_manager.getPrevEvents(.write) orelse return;
            try cl.event.wait_for_many(prev_events);
        }
    };
}
