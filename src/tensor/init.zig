const std = @import("std");

const core = @import("core");
const CommandQueue = core.CommandQueue;

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;


pub fn initTensorProperties(
    comptime T: type,
    comptime is_complex: bool,
    comptime type_id: comptime_int,
    command_queues: []const CommandQueue,
    arena_allocator: std.mem.Allocator,
    tensor: *Tensor(T),
    shape: []const u64,
    config_vectors_enabled: bool,
) TensorErrors!usize {
    var vectors_enabled = (!is_complex and config_vectors_enabled);
    var vector_width: u64 = 1;
    if (vectors_enabled) {
        for (command_queues) |cmd| {
            const cw: u64 = @intCast(cmd.vector_widths[type_id]);
            vector_width = @max(cw, vector_width);
        }
        vectors_enabled &= vector_width > 1;
    }

    tensor.flags.vectors_enabled = vectors_enabled;

    const vl_shape = try arena_allocator.dupe(u64, shape);
    tensor.dimensions.vl_shape = vl_shape;

    const ndim = shape.len;

    const last_element_index = ndim - 1;
    const penultimate_element_index = last_element_index -| 1;

    var number_of_elements_without_padding: u64 = 1;
    for (shape[0..penultimate_element_index]) |e| number_of_elements_without_padding *= e;
    const depth: usize = number_of_elements_without_padding;

    const penultimate_size = if (ndim >= 2) shape[penultimate_element_index] else 1;
    const last_size = shape[last_element_index];

    var padded_penultimate_size = penultimate_size;
    if (!is_complex and vectors_enabled and vector_width > 1) {
        const remainder = @mod(padded_penultimate_size, vector_width);
        if (remainder > 0) {
            padded_penultimate_size += vector_width - remainder;
        }
    }
    const penultimate_size_for_vectors = padded_penultimate_size / vector_width;
    padded_penultimate_size = (penultimate_size_for_vectors + penultimate_size_for_vectors % 2) * vector_width;

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

    const antepenultimate_element_index = penultimate_element_index -| 1;
    const pitches = tensor.dimensions.pitches;
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

    return number_of_elements;
}

