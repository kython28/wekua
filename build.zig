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
    });

    const core_module = b.addModule("core", .{
        .root_source_file = b.path("src/core/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    core_module.addImport("opencl", opencl_module);

    const tensor_module = b.addModule("tensor", .{
        .root_source_file = b.path("src/tensor/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    tensor_module.addImport("opencl", opencl_module);
    tensor_module.addImport("core", core_module);
    tensor_module.addImport("utils", utils_module);

    const blas_module = b.addModule("blas", .{
        .root_source_file = b.path("src/blas/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    blas_module.addImport("opencl", opencl_module);
    blas_module.addImport("core", core_module);
    blas_module.addImport("utils", utils_module);
    blas_module.addImport("tensor", tensor_module);

    const math_module = b.addModule("math", .{
        .root_source_file = b.path("src/math/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    math_module.addImport("opencl", opencl_module);
    math_module.addImport("core", core_module);
    math_module.addImport("utils", utils_module);
    math_module.addImport("tensor", tensor_module);

    const nn_module = b.addModule("nn", .{
        .root_source_file = b.path("src/nn/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    nn_module.addImport("opencl", opencl_module);
    nn_module.addImport("core", core_module);
    nn_module.addImport("utils", utils_module);
    nn_module.addImport("tensor", tensor_module);
    nn_module.addImport("blas", blas_module);
    nn_module.addImport("math", math_module);

    const wekua_module = b.addModule("wekua", .{
        .root_source_file = b.path("src/wekua.zig"),
        .target = target,
        .optimize = optimize,
    });
    wekua_module.addImport("opencl", opencl_module);
    wekua_module.addImport("core", core_module);
    wekua_module.addImport("tensor", tensor_module);
    wekua_module.addImport("utils", utils_module);
    wekua_module.addImport("blas", blas_module);
    wekua_module.addImport("math", math_module);
    wekua_module.addImport("nn", nn_module);

    const test_step = b.step("test", "Run unit tests");
    const run_check_step = b.step("check", "ZLS");

    const modules_to_test = .{
        .{core_module, "core"},
        .{tensor_module, "tensor"},
        .{blas_module, "blas"},
        .{math_module, "math"},
        .{nn_module, "nn"},
    };
    inline for (modules_to_test) |module_to_test| {
        const module = module_to_test[0];
        const name = module_to_test[1];

        const test_compilation_step = b.addTest(.{
            .root_module = module,
            .use_llvm = true,
            .name = name,
        });

        const run = b.addRunArtifact(test_compilation_step);
        run.has_side_effects = true;

        test_step.dependOn(&run.step);
        run_check_step.dependOn(&test_compilation_step.step);
    }

    const benchmark_module = b.addModule("benchmark", .{
        .root_source_file = b.path("benchmark/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    benchmark_module.addImport("wekua", wekua_module);

    const benchmark = b.addExecutable(.{
        .root_module = benchmark_module,
        .name = "benchmark",
        .use_llvm = true,
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
}
