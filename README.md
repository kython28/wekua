![](https://github.com/kython28/wekua/raw/master/media/wekua.png)

# Linear Algebra and Deep Learning Zig library with GPGPU and heterogeneous computing 

**Wekua** is a lightweight and optimized library that allows you to build and run deep learning models using GPGPU and heterogeneous computing with OpenCL.

Thanks to OpenCL, Wekua offers the flexibility to build and run deep learning models on any processing chip that supports OpenCL. Whether you have an AMD, NVIDIA, or Intel GPU, or even just a CPU, you can run them seamlessly. Additionally, **you can combine the power of all the processing chips available in your machine to execute your deep learning models or any tensor operations, making the most of your hardware resources**.

## Getting Started

First off, you must install the opencl runtime to use your hardware in wekua. For more info: https://wiki.archlinux.org/index.php/GPGPU#OpenCL_Runtime

Once the OpenCL runtime is installed, all that remains is to install wekua on your project. To include it, follow these steps:

1. **Edit your `build.zig.zon`:**
Add the `wekua` dependency to your `build.zig.zon` file, similar to the following:
```zig
.{
    // ......
    .dependencies = .{
        .wekua = .{
            .url = "https://github.com/kython28/wekua/archive/refs/tags/v{version}.tar.gz",
        },
    },
    // .....
}
```

2. **Edit your `build.zig`:**
Configure your `build.zig` file to include the `wekua` module:
```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const package = b.dependency("wekua", .{
        .target = target,
        .optimize = optimize
    });

    const module = package.module("wekua");

    const app = b.addExecutable(.{
        .name = "test-wekua",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize
    });

    app.root_module.addImport("wekua", module);

    b.installArtifact(app);
}
```

3. **Build the project:**
Now you can build your project:
```bash
zig build
```
This setup will allow you to use the `wekua` module in your project.


# How to use?

Now let's see a simple example of how to use Wekua to build a neural network. Let's try to solve the XOR problem, which is a classic problem in neural networks.


# ⚠️ Note

Currently, Wekua is available in two versions: one in C and another in Zig. The Zig version is under active development. If you want to try it out, you can compile it directly from the build.zig file and run its tests without any issues.

The C version is fully functional and stable at the moment. However, it will be deprecated once the Zig version covers all functionalities and offers the same level of stability.
