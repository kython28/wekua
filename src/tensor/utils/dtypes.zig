const std = @import("std");
const cl = @import("opencl");

const wContext = @import("../../core/context.zig").wContext;
const wLinkedList = @import("../../utils/linked_list.zig");

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

pub const number_of_dtypes: usize = @typeInfo(wTensorDtype).@"enum".fields.len;

pub const wScalar = union(wTensorDtype) {
    int8: i8,
    uint8: u8,

    int16: i16,
    uint16: u16,

    int32: i32,
    uint32: u32,

    int64: i64,
    uint64: u64,

    float32: f32,
    float64: f64
};

pub fn initialize_scalar(dtype: wTensorDtype, init_value: anytype) wScalar {
    return switch (dtype) {
        .int8 => .{.int8 = init_value},
        .uint8 => .{.uint8 = init_value},
        .int16 => .{.int16 = init_value},
        .uint16 => .{.uint16 = init_value},
        .int32 => .{.int32 = init_value},
        .uint32 => .{.uint32 = init_value},
        .int64 => .{.int64 = init_value},
        .uint64 => .{.uint64 = init_value},
        .float32 => .{.float32 = init_value},
        .float64 => .{.float64 = init_value}
    };
}

pub fn create_scalar(value: anytype) wScalar {
    const T = @TypeOf(value);
    const dtype = comptime get_wekua_dtype_from_zig_type(T);
    var new_scalar = initialize_scalar(dtype, undefined);
    const tensor_dtype_fields = @typeInfo(wTensorDtype).@"enum".fields;
    inline for (tensor_dtype_fields) |field| {
        if (@field(wTensorDtype, field.name) == dtype) {
            @field(new_scalar, field.name) = value;
        }
    }
    return new_scalar;
}

pub const wCreateTensorConfig = struct {
    dtype: wTensorDtype,
    cl_mem_flags: cl.buffer.cl_mem_flags = @intFromEnum(cl.buffer.enums.mem_flags.read_write),
    host_ptr: ?*anyopaque = null,
    is_complex: bool = false,
    vectors_enabled: bool = true
};

pub const _w_tensor = struct {
    context: wContext,

    buffer: cl.buffer.cl_mem,

    shape: []u64,
    vl_shape: []u64,

    number_of_elements: u64,
    number_of_elements_without_padding: u64,
    number_of_vectors: u64,

    row_pitch: u64,
    pitchs: []u64,
    pitchs_buffer: cl.buffer.cl_mem,

    row_pitch_for_vectors: u64,

    size: usize,

    dtype: wTensorDtype,
    is_complex: bool,
    vectors_enabled: bool,

    shape_like_matrix: [2]u64,
    shape_like_matrix_without_vectors: [2]u64,

    work_item_for_all_elements: []u64,
    work_item_for_all_vectors: []u64,
    work_items_like_matrix: [][2]u64,
    work_items_like_matrix_without_vectors: [][2]u64,

    mutex: std.Thread.Mutex,
    condition: std.Thread.Condition,
    events: wLinkedList
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

pub fn get_wekua_dtype_from_zig_type(comptime T: type) wTensorDtype {
    if (!@inComptime()) {
        @panic("This function must run in comptime");
    }

    return switch (T) {
        i8 => .int8,
        u8 => .uint8,
        i16 => .int16,
        u16 => .uint16,
        i32 => .int32,
        u32 => .uint32,
        i64 => .int64,
        u64 => .uint64,
        f32 => .float32,
        f64 => .float64,
        else => @panic("Type not supported")
    };
}
