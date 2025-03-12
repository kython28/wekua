pub const wLinkedList = @import("linked_list.zig");
pub const wQueue = @import("queue.zig");
pub usingnamespace @import("flatten_unflatten_indices.zig");

const std = @import("std");

pub fn calculate_work_items(global_work_items: []const u64, local_work_items: []u64, max_work_group_size: u64) void {
    const max_per_cu: u64 = @intFromFloat(
        std.math.pow(
            f64, @as(f64, @floatFromInt(max_work_group_size)), 1.0 / @as(f64, @floatFromInt(global_work_items.len))
        )
    );

    for (global_work_items, local_work_items) |g, *l| {
        if (g < max_per_cu) {
            l.* = g;
            continue;
        }

        var li: u64 = max_per_cu;
        while (@mod(g, li) != 0) {
            li -= 1;
        }
        l.* = li;
    }
}

pub fn release_temporal_resource_callback(allocator: std.mem.Allocator, user_data: ?*anyopaque) void {
    const ptr = @intFromPtr(user_data.?);
    const size: *usize = @ptrFromInt(ptr);
    allocator.free(@as([*]u8, @ptrFromInt(ptr))[0..size.*]);
}
