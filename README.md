![](https://github.com/kython28/wekua/raw/master/media/wekua.png)

**Wekua** is a lightweight and optimized library that allows you to build and run deep learning models using GPGPU and heterogeneous computing with OpenCL.

Thanks to OpenCL, Wekua offers the flexibility to build and run deep learning models on any processing chip that supports OpenCL. Whether you have an AMD, NVIDIA, or Intel GPU, or even just a CPU, you can run them seamlessly. Additionally, **you can combine the power of all the processing chips available in your machine to execute your deep learning models or any tensor operations, making the most of your hardware resources**.

# How to install?
First off, you must install the opencl runtime to use your processing chips in wekua. For more info: https://wiki.archlinux.org/index.php/GPGPU#OpenCL_Runtime

Well, once the OpenCL runtime is installed, all that remains is to install wekua. Let's first clone the repository:

```sh
git clone https://github.com/kython28/wekua
```

Now we just have to:

```sh
cd wekua
make
sudo make install
make clean
```

# How to use? (Comming soon)

# ⚠️ Note

Currently, Wekua is available in two versions: one in C and another in Zig. The Zig version is under active development. If you want to try it out, you can compile it directly from the build.zig file and run its tests without any issues.

The C version is fully functional and stable at the moment. However, it will be deprecated once the Zig version covers all functionalities and offers the same level of stability.
