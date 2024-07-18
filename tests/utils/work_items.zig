const std = @import("std");
const wekua = @import("wekua");

test {
    const global_work_items: []const u64 = &[_]u64{20, 30, 200, 1000, 1};
    const expected_values: []const u64 = &[_]u64{
        20,
        20, 15,
        10, 10, 10,
        2, 2, 2, 2,
        5, 5, 5, 5, 1
    };
    var local_work_items: [1 + 2 + 3 + 4 + 5]u64 = undefined;

    try wekua.utils.work_items.get(global_work_items[0..1], local_work_items[0..1], 1024);
    try wekua.utils.work_items.get(global_work_items[0..2], local_work_items[1..3], 512);
    try wekua.utils.work_items.get(global_work_items[0..3], local_work_items[3..6], 2024);
    try wekua.utils.work_items.get(global_work_items[0..4], local_work_items[6..10], 64);
    try wekua.utils.work_items.get(global_work_items[0..5], local_work_items[10..15], 4096);

    try std.testing.expectEqualSlices(u64, local_work_items[0..1], expected_values[0..1]);
    try std.testing.expectEqualSlices(u64, local_work_items[1..3], expected_values[1..3]);
    try std.testing.expectEqualSlices(u64, local_work_items[3..6], expected_values[3..6]);
    try std.testing.expectEqualSlices(u64, local_work_items[6..10], expected_values[6..10]);
    try std.testing.expectEqualSlices(u64, local_work_items[10..15], expected_values[10..15]);

}
