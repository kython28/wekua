const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const opencl_package = b.dependency("zig-opencl", .{
        .target = target,
        .optimize = optimize
    });
    const opencl_module = opencl_package.module("opencl");

    const wekua_module = b.addModule("wekua", .{
        .root_source_file = b.path("src/wekua.zig"),
        .target = target,
        .optimize = optimize
    });
    wekua_module.addImport("opencl", opencl_module);

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("tests/wekua.zig"),
        .target = target,
        .optimize = optimize
    });
    lib_unit_tests.root_module.addImport("wekua", wekua_module);
    lib_unit_tests.root_module.addImport("opencl", opencl_module);

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    run_lib_unit_tests.has_side_effects = true;

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
