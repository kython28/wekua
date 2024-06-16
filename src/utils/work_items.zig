const std = @import("std");

pub fn get(global_work_items: []u64, local_work_items: []u64, max_work_group_size: u64) error{LengthsNotMatching}!void {
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
