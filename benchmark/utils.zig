pub const report = struct {
    avg_times_per_bactch: [12]f64 = .{0} ** 12,
    name: []const u8
};

pub const wekua_c = @cImport({
    @cInclude("wekua/wekua.h");
    @cInclude("wekua/matrix.h");
});

pub const openblas = @cImport({
    @cInclude("cblas.h");
});
