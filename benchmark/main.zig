const std = @import("std");
const builtin = @import("builtin");

const axpy = @import("axpy.zig");
const gemm = @import("gemm.zig");

const modules = .{
    // axpy,
    gemm,
};

const col_width = 12;
const name_col_width = 22;

fn printHLine(stdout: anytype, ncols: usize, comptime left: []const u8, comptime mid: []const u8, comptime right: []const u8) !void {
    try stdout.writeAll(left);
    for (0..name_col_width) |_| try stdout.writeAll("\u{2500}");
    for (0..ncols) |_| {
        try stdout.writeAll(mid);
        for (0..col_width) |_| try stdout.writeAll("\u{2500}");
    }
    try stdout.writeAll(right);
    try stdout.writeAll("\n");
}

pub fn main() !void {
    var stdout_file = std.fs.File.stdout();
    var writer = stdout_file.writer(&.{});
    const stdout = &writer.interface;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        _ = gpa.deinit();
    }

    const allocator = blk: {
        if (builtin.mode == .Debug) {
            break :blk gpa.allocator();
        }
        break :blk std.heap.c_allocator;
    };

    try stdout.print("\n", .{});
    try stdout.print("  wekua benchmark suite ({s})\n", .{@tagName(builtin.mode)});
    try stdout.print("\n", .{});

    inline for (modules) |module| {
        const reports = try module.run_benchmark(allocator);
        defer allocator.free(reports);

        const ncols = reports[0].avg_times_per_batch.len;
        const starting_point: u64 = module.starting_point;

        // Title
        try stdout.print("  {s}  dtype={s}  iters={d}\n", .{ module.name, module.dtype, module.niterations });
        try stdout.print("\n", .{});

        // Top border
        try printHLine(stdout, ncols, "  \u{250C}", "\u{252C}", "\u{2510}");

        // Header row
        try stdout.print("  \u{2502}{s: ^" ++ comptimeFmt(name_col_width) ++ "}", .{"Library"});
        for (0..ncols) |i| {
            try stdout.print("\u{2502}{d: ^" ++ comptimeFmt(col_width) ++ "}", .{starting_point * std.math.pow(u64, 2, i)});
        }
        try stdout.print("\u{2502}\n", .{});

        // Header separator
        try printHLine(stdout, ncols, "  \u{251C}", "\u{253C}", "\u{2524}");

        // Data rows
        for (reports) |report| {
            try stdout.print("  \u{2502}{s: <" ++ comptimeFmt(name_col_width) ++ "}", .{report.name});
            for (report.avg_times_per_batch) |time| {
                if (time == 0) {
                    try stdout.print("\u{2502}{s: ^" ++ comptimeFmt(col_width) ++ "}", .{"-"});
                } else {
                    try stdout.print("\u{2502}{d: ^" ++ comptimeFmt(col_width) ++ ".3}", .{time});
                }
            }
            try stdout.print("\u{2502}\n", .{});
        }

        // Bottom border
        try printHLine(stdout, ncols, "  \u{2514}", "\u{2534}", "\u{2518}");

        // Units note
        try stdout.print("  times in ms\n", .{});
        try stdout.print("\n", .{});
    }
}

fn comptimeFmt(comptime width: usize) []const u8 {
    return std.fmt.comptimePrint("{d}", .{width});
}
