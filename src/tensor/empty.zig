const std = @import("std");
const cl = @import("opencl");

const wContext = @import("../core/context.zig").wContext;
const work_items = @import("../utils/work_items.zig");
const linked_list = @import("../utils/linked_list.zig");
const w_event = @import("event.zig");

pub const wTensorDtype = enum (u8) {
    int8 = 0,
    uint8 = 1,

    int16 = 2,
    uint16 = 3,

    int32 = 4,
    uint32 = 5,

    int64 = 6,
    uint64 = 7,

    float32 = 8,
    float64 = 9
};

pub const wCreateTensorConfig = struct {
    dtype: wTensorDtype,
    cl_mem_flags: cl.buffer.cl_mem_flags = @intFromEnum(cl.buffer.enums.mem_flags.read_write),
    host_ptr: ?*anyopaque = null,
    is_complex: bool = false,
    vectors_enabled: bool = true
};

const _w_tensor = struct {
    context: wContext,

    buffer: cl.buffer.cl_mem,

    shape: []u64,
    vl_shape: []u64,

    number_of_elements: u64,
    col_pitch: u64,
    col_pitch_for_vectors: u64,
    size: usize,

    dtype: wTensorDtype,
    is_complex: bool,
    vectors_enabled: bool,

    work_item_for_all_elements: []u64,
    work_items_like_matrix: [][2]u64,
    work_items_like_matrix_without_vectors: [][2]u64,

    mutex: *std.Thread.Mutex,
    events: linked_list.wLinkedList
};

pub const wTensor = *_w_tensor;

pub fn get_dtype_size(dtype: wTensorDtype) usize {
    return switch (dtype) {
        .int8 => @sizeOf(i8),
        .uint8 => @sizeOf(u8),
        .int16 => @sizeOf(i16),
        .uint16 => @sizeOf(u16),
        .int32 => @sizeOf(i32),
        .uint32 => @sizeOf(u32),
        .int64 => @sizeOf(i64),
        .uint64 => @sizeOf(u64),
        .float32 => @sizeOf(f32),
        .float64 => @sizeOf(f64)
    };
}

pub fn empty(context: wContext, shape: []const u64, config: wCreateTensorConfig) !wTensor {
    const allocator = context.allocator;
    const command_queues = context.command_queues;

    const tensor: wTensor = try allocator.create(_w_tensor);
    errdefer allocator.destroy(tensor);

    tensor.context = context;
    tensor.events = try linked_list.create(allocator);
    const mutex = try allocator.create(std.Thread.Mutex);
    errdefer allocator.destroy(mutex);

    mutex.* = std.Thread.Mutex{};
    tensor.mutex = mutex;

    tensor.shape = try allocator.alloc(u64, shape.len);
    errdefer allocator.free(tensor.shape);
    std.mem.copyForwards(u64, tensor.shape, shape);

    const dtype = config.dtype;
    const is_complex = config.is_complex;
    const vectors_enabled = if (is_complex) false else config.vectors_enabled;

    tensor.dtype = dtype;
    tensor.is_complex = is_complex;
    tensor.vectors_enabled = vectors_enabled;

    var vector_width: u64 = 1;
    if (vectors_enabled) {
        for (command_queues) |cmd| {
            const cw: u64 = @intCast(cmd.vector_widths[@intFromEnum(dtype)]);
            vector_width = @max(cw, vector_width);
        }
    }

    const vl_shape = try allocator.alloc(u64, shape.len);
    tensor.vl_shape = vl_shape;
    errdefer allocator.free(vl_shape);
    std.mem.copyForwards(u64, vl_shape, shape);

    const last_element_index = shape.len - 1;
    var col_pitch: u64 = shape[last_element_index];
    col_pitch += vector_width - @mod(col_pitch, vector_width);
    if (is_complex) col_pitch *= 2;
    const col_pitch_for_vectors = col_pitch / vector_width;
    tensor.col_pitch = col_pitch;
    tensor.col_pitch_for_vectors = col_pitch_for_vectors;

    vl_shape[last_element_index] = col_pitch_for_vectors;

    var number_of_elements: u64 = 1;
    for (shape[0..last_element_index]) |e| number_of_elements *= e;
    number_of_elements *= col_pitch;
    tensor.number_of_elements = number_of_elements;

    const work_item_for_all_elements: []u64 = try allocator.alloc(u64, command_queues.len);
    errdefer allocator.free(work_item_for_all_elements);

    const work_items_like_matrix: [][2]u64 = try allocator.alloc([2]u64, command_queues.len);
    errdefer allocator.free(work_items_like_matrix);

    const work_items_like_matrix_without_vectors: [][2]u64 = try allocator.alloc([2]u64, command_queues.len);
    errdefer allocator.free(work_items_like_matrix_without_vectors);

    tensor.work_item_for_all_elements = work_item_for_all_elements;
    tensor.work_items_like_matrix = work_items_like_matrix;
    tensor.work_items_like_matrix_without_vectors = work_items_like_matrix_without_vectors;

    for (
        command_queues, work_item_for_all_elements,
        work_items_like_matrix,
        work_items_like_matrix_without_vectors
    ) |cmd, *wa, *wmv, *wm| {
        try work_items.get(
            @as([*]u64, @ptrCast(&number_of_elements))[0..1],
            @as([*]u64, @ptrCast(wa))[0..1],
            cmd.max_work_group_size
        );

        const rows = number_of_elements / col_pitch;
        try work_items.get(
            &[2]u64{rows, col_pitch_for_vectors},
            wmv,
            cmd.max_work_group_size
        );

        try work_items.get(
            &[2]u64{rows, col_pitch},
            wm,
            cmd.max_work_group_size
        );
    }

    const size: usize = number_of_elements * get_dtype_size(dtype);
    tensor.size = size;

    tensor.buffer = try cl.buffer.create(context.ctx, config.cl_mem_flags, size, config.host_ptr);
    return tensor;
}

pub fn release(tensor: wTensor) void {
    const allocator = tensor.context.allocator;

    const events = tensor.events;
    const last_event = w_event.acquire_tensor(tensor);

    events.first = null;
    if (last_event) |event| {
        cl.event.wait(event) catch unreachable;

        const last_node = events.last.?;
        w_event.release_tensor_event(@alignCast(@ptrCast(last_node.data.?)));
        allocator.destroy(last_node);

        events.last = null;
    }

    linked_list.release(events) catch unreachable;
    cl.buffer.release(tensor.buffer) catch unreachable;
    tensor.mutex.unlock();

    allocator.free(tensor.shape);
    allocator.free(tensor.vl_shape);
    allocator.free(tensor.work_item_for_all_elements);
    allocator.free(tensor.work_items_like_matrix);
    allocator.free(tensor.work_items_like_matrix_without_vectors);
    allocator.destroy(tensor.mutex);
    allocator.destroy(tensor);
}
