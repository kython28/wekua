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

    const module_name_to_test = b.option([]const u8, "module_to_test", "Name of the module to test");

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

        if (module_name_to_test == null or std.mem.eql(u8, name, module_name_to_test.?)) {
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
    }

    // Examples
    const example_name = b.option([]const u8, "example", "Name of the example to run");
    const run_example_step = b.step("run_example", "Run an example (use -Dexample=<name>)");

    if (example_name) |name| {
        const example_path = b.fmt("examples/{s}.zig", .{name});

        const example_module = b.addModule(name, .{
            .root_source_file = b.path(example_path),
            .target = target,
            .optimize = optimize,
        });
        example_module.addImport("wekua", wekua_module);

        const example_exe = b.addExecutable(.{
            .root_module = example_module,
            .name = name,
            .use_llvm = true,
        });

        const install_example = b.addInstallArtifact(example_exe, .{});

        const run_example = b.addRunArtifact(example_exe);
        run_example.has_side_effects = true;

        run_example_step.dependOn(&install_example.step);
        run_example_step.dependOn(&run_example.step);
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
