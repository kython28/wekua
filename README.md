![](https://github.com/kython28/wekua/raw/master/media/wekua.png)
**Wekua** is a lightweight and optimized library where you can build deep learning models and run them with GPGPU and heterogeneous computing using OpenCL.

Thanks to OpenCL, you will have the possibility to build and run your deep learning models in Wekua with any processing chip that supports OpenCL. It doesn't matter if you have an AMD, NVIDIA, or Intel GPU, you can run them the same way, and it doesn't matter if you only have one CPU. **You can even combine the power of all the processing chips you have in your machine**.

# How to install?
First off, you must install the opencl runtime to use your processing chips in wekua. For more info: https://wiki.archlinux.org/index.php/GPGPU#OpenCL_Runtime

Well, once the OpenCL runtime is installed, all that remains is to install wekua. Let's first clone the repository:

    git clone https://github.com/kython28/wekua

Now we just have to:

    cd wekua
    make
    sudo make install
    make clean

# How to use? (Comming soon)