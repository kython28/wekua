pub const report = struct {
    avg_times_per_bactch: [10]f64 = .{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    name: []const u8
};

pub const wekua_c = @cImport({
    @cInclude("wekua/wekua.h");
    @cInclude("wekua/matrix.h");
});
