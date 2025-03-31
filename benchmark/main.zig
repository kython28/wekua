const std = @import("std");
const builtin = @import("builtin");

const axpy = @import("axpy.zig");
const gemm = @import("gemm.zig");

const modules = .{
    gemm
};

pub fn main() !void {
    const stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();

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

    inline for (modules) |module| {
        try stdout.print("-----------------------------------------------------------\n", .{});
        try stdout.print("Starting benchmark test for {s}\n", .{module.name});

        const reports = try module.run_benchmark(allocator);
        defer allocator.free(reports);

        const starting_point: u64 = module.starting_point;
        try stdout.print("{s: ^15} | ", .{"Library"});
        for (0..reports[0].avg_times_per_bactch.len) |i| {
            try stdout.print("{d: ^10}", .{starting_point * std.math.pow(u64, 2, i)});
        }
        try stdout.print("\n", .{});
        try stdout.print("------------------", .{});
        for (0..reports[0].avg_times_per_bactch.len) |_| {
            try stdout.print("----------", .{});
        }
        try stdout.print("\n", .{});

        // try stdout.print("{s: ^15} | {d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}\n",
        //     .{"Library"} ++ sizes
        // );
        for (reports) |report| {
            try stdout.print("{s: <15} | ", .{report.name});
            for (report.avg_times_per_bactch) |time| {
                try stdout.print("{d: ^10.3}", .{time});
            }
            try stdout.print("\n", .{});
        }
    }
}
