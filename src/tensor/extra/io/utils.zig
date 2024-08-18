const std = @import("std");

const dtypes = @import("../../utils/dtypes.zig");
const wTensorDtype = dtypes.wTensorDtype;

const w_errors = @import("../../utils/errors.zig");

pub fn check_buffer_type(dtype: wTensorDtype, buffer: anytype) !void {
    const buffer_type_info = @typeInfo(@TypeOf(buffer));
    if (buffer_type_info != .Pointer) {
        @panic("Buffer have to be a Slice");
    }

    if (buffer_type_info.Pointer.size != .Slice) {
        @panic("Buffer have to be a Slice");
    }

    const buffer_dtype = dtypes.get_wekua_dtype_from_zig_type(buffer_type_info.Pointer.child);
    if (buffer_dtype != dtype) {
        return w_errors.errors.InvalidBuffer;
    }
}

pub fn signal_condition_callback(_: std.mem.Allocator, user_data: ?*anyopaque) void {
    const cond: *std.Thread.Condition = @alignCast(@ptrCast(user_data.?));
    cond.signal();
}

