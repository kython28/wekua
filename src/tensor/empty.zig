const std = @import("std");
const cl = @import("opencl");

const wContext = @import("../core/context.zig").wContext;
const work_items = @import("../utils/work_items.zig");
const linked_list = @import("../utils/linked_list.zig");
const w_event = @import("utils/event.zig");

const wMutex = @import("../utils/mutex.zig").wMutex;
const wCondition = @import("../utils/condition.zig").wCondition;

const dtypes = @import("utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wCreateTensorConfig = dtypes.wCreateTensorConfig;

pub fn empty(context: wContext, shape: []const u64, config: wCreateTensorConfig) !wTensor {
    const allocator = context.allocator;
    const command_queues = context.command_queues;

    const tensor: wTensor = try allocator.create(dtypes._w_tensor);
    errdefer allocator.destroy(tensor);

    tensor.context = context;
    tensor.events = try linked_list.create(allocator);
    const mutex = try allocator.create(wMutex);
    mutex.* = wMutex{};
    errdefer {
        mutex.destroy();
        allocator.destroy(mutex);
    }
    tensor.mutex = mutex;

    const condition = try allocator.create(wCondition);
    condition.* = wCondition{};
    errdefer {
        condition.destroy();
        allocator.destroy(condition);
    }
    tensor.condition = condition;

    tensor.shape = try allocator.alloc(u64, shape.len);
    errdefer allocator.free(tensor.shape);
    @memcpy(tensor.shape, shape);

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
    @memcpy(vl_shape, shape);

    const last_element_index = shape.len - 1;
    var col_pitch: u64 = shape[last_element_index];
    if (vectors_enabled and vector_width > 1) {
        col_pitch += vector_width - @mod(col_pitch, vector_width);
    }
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

    const size: usize = number_of_elements * dtypes.get_dtype_size(dtype);
    tensor.size = size;

    tensor.buffer = try cl.buffer.create(context.ctx, config.cl_mem_flags, size, config.host_ptr);
    return tensor;
}

pub fn release(tensor: wTensor) void {
    const allocator = tensor.context.allocator;

    const events = tensor.events;
    events.first = null;

    tensor.mutex.lock();
    if (events.last) |last_node| {
        const tensor_event: w_event.wTensorEvent = @alignCast(@ptrCast(last_node.data.?));
        switch (tensor_event.event_type) {
            .write => {
                if (tensor_event.events_finalized != 1){
                    tensor.condition.wait(tensor.mutex);
                }
            },
            .read => {
                if (tensor_event.events_finalized != tensor_event.read_events.?.items.len) {
                    tensor.condition.wait(tensor.mutex);
                }
            }
        }
        w_event.release_tensor_event(tensor_event);
        allocator.destroy(last_node);
        events.last = null;
    }
    tensor.mutex.unlock();

    linked_list.release(events) catch unreachable;
    cl.buffer.release(tensor.buffer) catch unreachable;
    tensor.mutex.destroy();
    tensor.condition.destroy();

    allocator.free(tensor.shape);
    allocator.free(tensor.vl_shape);
    allocator.free(tensor.work_item_for_all_elements);
    allocator.free(tensor.work_items_like_matrix);
    allocator.free(tensor.work_items_like_matrix_without_vectors);
    allocator.destroy(tensor.mutex);
    allocator.destroy(tensor.condition);
    allocator.destroy(tensor);
}
