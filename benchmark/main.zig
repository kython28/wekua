const std = @import("std");
const builtin = @import("builtin");

const utils = @import("utils.zig");

const axpy = @import("axpy.zig");


const modules = .{
    axpy,
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

        try stdout.print("Showing results . . . (results in ms)\n", .{});

        try stdout.print("{s: ^15} | {d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}{d: ^10}\n", .{
            "Library", 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
        });
        for (reports) |report| {
            try stdout.print("{s: <15} | {d: ^10.3}{d: ^10.3}{d: ^10.3}{d: ^10.3}{d: ^10.3}{d: ^10.3}{d: ^10.3}{d: ^10.3}{d: ^10.3}{d: ^10.3}\n", .{
                report.name, report.avg_times_per_bactch[0], report.avg_times_per_bactch[1], report.avg_times_per_bactch[2],
                report.avg_times_per_bactch[3], report.avg_times_per_bactch[4], report.avg_times_per_bactch[5],
                report.avg_times_per_bactch[6], report.avg_times_per_bactch[7], report.avg_times_per_bactch[8],
                report.avg_times_per_bactch[9]
            });
        }
    }
}
