const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const opencl_package = b.dependency("zig-opencl", .{
        .target = target,
        .optimize = optimize,
    });
    const opencl_module = opencl_package.module("opencl");

    const utils_module = b.addModule("utils", .{
        .root_source_file = b.path("src/utils/utils.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = false,
    });

    const core_module = b.addModule("core", .{
        .root_source_file = b.path("src/core/main.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = false,
    });
    core_module.addImport("opencl", opencl_module);

    const tensor_module = b.addModule("tensor", .{
        .root_source_file = b.path("src/tensor/main.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = false,
    });
    tensor_module.addImport("opencl", opencl_module);
    tensor_module.addImport("core", core_module);
    tensor_module.addImport("utils", utils_module);

    const wekua_module = b.addModule("wekua", .{
        .root_source_file = b.path("src/wekua.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = false,
    });
    wekua_module.addImport("opencl", opencl_module);
    wekua_module.addImport("core", core_module);
    wekua_module.addImport("tensor", tensor_module);
    wekua_module.addImport("utils", utils_module);

    const core_test = b.addTest(.{
        .root_module = core_module,
    });

    const tensor_test = b.addTest(.{
        .root_module = tensor_module,
    });

    const run_core_test = b.addRunArtifact(core_test);
    const run_tensor_test = b.addRunArtifact(tensor_test);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_core_test.step);
    test_step.dependOn(&run_tensor_test.step);

    const benchmark_module = b.addModule("benchmark", .{
        .root_source_file = b.path("benchmark/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    benchmark_module.addImport("wekua", wekua_module);

    const benchmark = b.addExecutable(.{
        .root_module = benchmark_module,
        .name = "benchmark",
    });
    benchmark.addIncludePath(.{ .cwd_relative = "/usr/include/" });
    benchmark.linkSystemLibrary("wekua");
    benchmark.linkSystemLibrary("openblas");

    const install_benchmark = b.addInstallArtifact(benchmark, .{});

    const run_benchmark = b.addRunArtifact(benchmark);
    run_benchmark.has_side_effects = true;

    const run_benchmark_step = b.step("benchmark", "Run benchmark");
    run_benchmark_step.dependOn(&install_benchmark.step);
    run_benchmark_step.dependOn(&run_benchmark.step);

    const run_check_step = b.step("check", "ZLS");
    run_check_step.dependOn(&core_test.step);
    run_check_step.dependOn(&tensor_test.step);
}
