# CPU Feature Reference for HPC, Linear Algebra, FFT, ML, and Deep Learning

This document enumerates the CPU features that matter for high-performance numerical workloads, grouped by impact tier. Each entry lists the feature name, a description of what it does at the hardware level, the performance benefit, and the canonical Zig 0.16.0 enum path to query its presence on a resolved `std.Target`.

## Verification convention

All examples below assume you already have a `target: std.Target` obtained from `resolveTargetQuery` or `detectNativeCpuAndFeatures`. To check whether a feature is enabled, use:

```zig
const is_enabled = target.cpu.features.isEnabled(@intFromEnum(std.Target.<arch>.Feature.<name>));
```

The `<arch>` is `x86`, `x86_64`, or `aarch64` depending on the target.

---

## Tier 0 — Mandatory baseline

These features form the foundation. Any modern workload in linear algebra, FFT, ML, or DL implicitly assumes them. The performance penalty for missing any of them is severe.

### SSE 4.1 and SSE 4.2

- **Path:** `std.Target.x86.Feature.sse4_1`, `std.Target.x86.Feature.sse4_2`
- **What it does:** Adds 64-bit integer SIMD instructions, blend operations, `PMULLD`, `PMULDQ`, `CRC32`, and `CMPESTRM`. Doubles the integer throughput per cycle compared to SSE2.
- **Benefit:** Accelerates scalar kernels that have been vectorized with the minimum instruction set. Common in tight inner loops where register pressure is high and AVX/AVX2 would cause frequency throttling.

### AVX2

- **Path:** `std.Target.x86.Feature.avx2`
- **What it does:** Introduces 256-bit wide YMM registers for integer and floating-point SIMD, gather instructions, and per-lane shift/permute.
- **Benefit:** Roughly doubles vector throughput compared to SSE. Essential for packing, permutations, medium-sized GEMMs, FFT radix kernels, and quantized embedding operations.

### FMA (Fused Multiply-Add)

- **Path:** `std.Target.x86.Feature.fma`
- **What it does:** Adds `VFMADD*` instructions that perform `a * b + c` with a single rounding step.
- **Benefit:** Eliminates one rounding error and one memory fetch per multiply-add pair. Doubles the effective throughput of matrix multiplication and polynomial evaluation. This is the single largest single-instruction performance contribution for dense numerical code.

### F16C

- **Path:** `std.Target.x86.Feature.f16c`
- **What it does:** Provides `VCVTPH2PS` and `VCVTPS2PH`, vectorized conversion between FP16 and FP32.
- **Benefit:** Removes a bottleneck when mixing precisions. Useful anywhere an intermediate representation uses half precision for storage and single precision for computation.

### POPCNT

- **Path:** `std.Target.x86.Feature.popcnt`
- **What it does:** Counts the number of set bits in a register in a single cycle.
- **Benefit:** Useful for sparse representations, bit-packed weights, histogram aggregation, and certain hashing schemes. Trivial to detect and use.

### LZCNT and TZCNT

- **Path:** `std.Target.x86.Feature.lzcnt`, `std.Target.x86.Feature.tzcnt`
- **What it does:** Count leading or trailing zero bits in a single cycle.
- **Benefit:** Accelerates bit-scanning operations that are otherwise implemented as serial loops. Frequently used in quantization, sparse indexing, and attention masking.

### BMI1 and BMI2

- **Path:** `std.Target.x86.Feature.bmi1`, `std.Target.x86.Feature.bmi2`
- **What it does:** Adds `PDEP`, `PEXT`, `BLSI`, `BLSMSK`, `BLSR`, `ANDN`, conditional moves, and bit-field extraction.
- **Benefit:** `PDEP` and `PEXT` are invaluable for compressing and expanding bit fields in constant time. Critical for sparse attention, MoE routing, and lookup-table-based decoding.

---

## Tier 1 — High impact across the target workload set

These features unlock large jumps in throughput when present. They are not strictly required but are strongly preferred by any well-tuned library.

### AVX-512 foundation

The five-feature core below is what most libraries assume when they advertise "AVX-512 support".

#### AVX-512F (Foundation)

- **Path:** `std.Target.x86.Feature.avx512f`
- **What it does:** Introduces 32 new 512-bit ZMM registers, mask registers `k0–k7`, predicated execution, broadcast, and conflict detection.
- **Benefit:** Doubles SIMD width again. Enables the use of 16-wide FP32 and 8-wide FP64 lanes, which is the sweet spot for large GEMMs and FFTs.

#### AVX-512DQ (Doubleword and Quadword)

- **Path:** `std.Target.x86.Feature.avx512dq`
- **What it does:** Adds 64-bit integer permutes, FP conversions, and `VRANGEPS`.
- **Benefit:** Accelerates operations on 64-bit integers and improves precision control. Required for many optimized FFT twiddle routines.

#### AVX-512BW (Byte and Word)

- **Path:** `std.Target.x86.Feature.avx512bw`
- **What it does:** Extends 512-bit operations to byte and 16-bit word granularity.
- **Benefit:** Allows vectorization of byte-level processing such as histograms, tokenization, and certain compression routines.

#### AVX-512VL (Vector Length)

- **Path:** `std.Target.x86.Feature.avx512vl`
- **What it does:** Allows AVX-512 instructions to operate on 128-bit and 256-bit registers as well as the full 512-bit width.
- **Benefit:** Provides flexibility to mix vector widths within a single program. Crucial for masking and tail handling.

#### AVX-512CD (Conflict Detection)

- **Path:** `std.Target.x86.Feature.avx512cd`
- **What it does:** Adds `VPCONFLICTD`, `VPCONFLICTQ`, `VPLZCNTD`, `VPLZCNTQ`.
- **Benefit:** Speeds up scatter/gather operations and certain reduction patterns.

### AVX-512 VNNI (Vector Neural Network Instructions)

- **Path:** `std.Target.x86.Feature.avx512vnni`
- **What it does:** Introduces `VPDPBUSD`, `VPDPBUSDS`, `VPDPWSSD` and friends, performing signed/unsigned INT8 dot products with FP32 accumulation in 512-bit registers.
- **Benefit:** Accelerates quantized inner products. Multiplies the throughput of INT8 matrix multiplication compared to AVX2 VNNI emulation.

### AVX-512 BF16

- **Path:** `std.Target.x86.Feature.avx512bf16`
- **What it does:** Adds `VDPBF16PS`, performing BF16 dot products with FP32 accumulation.
- **Benefit:** Provides a high-throughput path for BF16 inner products without upcasting to FP32. Reduces register pressure and improves compute density.

### AVX-512 FP16

- **Path:** `std.Target.x86.Feature.avx512fp16`
- **What it does:** Adds native FP16 arithmetic in 512-bit registers.
- **Benefit:** Doubles the effective throughput of FP16 GEMMs and convolutions when compared to FP16C conversion paths. Eliminates the cost of converting between precisions.

### AMX (Advanced Matrix Extensions)

AMX introduces a 2D tile register file that operates on tiles of up to 16 rows of 8 64-bit elements. It is not a vector unit but a tile-based matrix engine.

#### AMX-TILE

- **Path:** `std.Target.x86.Feature.amx_tile`
- **What it does:** Introduces the tile register file and load/store instructions.
- **Benefit:** Enables tile-based matrix multiplication as a primitive.

#### AMX-INT8

- **Path:** `std.Target.x86.Feature.amx_int8`
- **What it does:** Adds `TDPBSSD`, `TDPBSUD`, `TDPBUSD`, `TDPBUUD`, performing INT8 dot products on tiles with INT32 accumulation.
- **Benefit:** Provides hardware acceleration for INT8 GEMMs that no amount of vector instructions can match.

#### AMX-BF16

- **Path:** `std.Target.x86.Feature.amx_bf16`
- **What it does:** Adds `TDPBF16PS`, performing BF16 dot products on tiles with FP32 accumulation.
- **Benefit:** Hardware path for BF16 GEMMs. Delivers order-of-magnitude gains over vectorized BF16 implementations on very large matrices.

### AVX VNNI

- **Path:** `std.Target.x86.Feature.avx_vnni`
- **What it does:** Brings `VPDPBUSD` and related INT8 dot products to 256-bit AVX2 registers.
- **Benefit:** Provides VNNI functionality on hardware that lacks full AVX-512 but has AVX2. Useful for 256-bit quantized GEMMs on Ice Lake-SP and Tiger Lake when AVX-512 has been fused off.

### GFNI (Galois Field New Instructions)

- **Path:** `std.Target.x86.Feature.gfni`
- **What it does:** Adds `GF2P8AFFINEINVQB`, `GF2P8AFFINEQB`, `GF2P8MULB`.
- **Benefit:** Accelerates Galois-field arithmetic used in certain algebraic codes and in graph-style algorithms. Can be repurposed for efficient bit-matrix multiplication.

### CLFLUSHOPT

- **Path:** `std.Target.x86.Feature.clflushopt`
- **What it does:** Optimized cache line flush with relaxed ordering. A subsequent store is not ordered with respect to the flush.
- **Benefit:** Faster cache-line invalidation when prefetching streams manually.

### CLWB

- **Path:** `std.Target.x86.Feature.clwb`
- **What it does:** Cache line write-back without eviction. Marks the line as least-recently-used without flushing it to memory.
- **Benefit:** Enables persistent caching of large working sets. Useful when a block will be reused after eviction if it remains in cache.

### MOVNTDQA

- **Path:** `std.Target.x86.Feature.movntdqa`
- **What it does:** Non-temporal aligned load from memory that bypasses the cache hierarchy.
- **Benefit:** Reads streaming data without polluting the cache. Critical for in-place transforms and large structured loads.

### Non-temporal stores

- **Path:** `std.Target.x86.Feature.movntps`, `std.Target.x86.Feature.movntpd`, `std.Target.x86.Feature.movntdq`
- **What it does:** Store instructions that bypass the cache, writing directly to memory.
- **Benefit:** Avoids polluting the cache when producing streaming output. Reduces write-combining overhead.

### RDRAND and RDSEED

- **Path:** `std.Target.x86.Feature.rdrand`, `std.Target.x86.Feature.rdseed`
- **What it does:** Hardware random number generation, with `RDSEED` providing a true entropy source and `RDRAND` providing a conditioned stream.
- **Benefit:** Cheap, high-quality random numbers for initialization and stochastic operations.

### FSGSBASE

- **Path:** `std.Target.x86.Feature.fsgsbase`
- **What it does:** Allows direct read and write of the FS and GS segment bases without syscalls.
- **Benefit:** Reduces TLS access overhead. Particularly valuable when running many threads.

### XSAVE family

- **Path:** `std.Target.x86.Feature.xsave`, `std.Target.x86.Feature.xsaveopt`, `std.Target.x86.Feature.xsavec`, `std.Target.x86.Feature.xsaves`
- **What it does:** Manages the extended register state (SSE, AVX, AVX-512, AMX) during context switches.
- **Benefit:** Required for fast context switches when the extended state is large. With AMX, `XSAVE` is mandatory because the tile state is too large to spill on every switch without it.

### SHA-NI

- **Path:** `std.Target.x86.Feature.sha`
- **What it does:** Hardware-accelerated SHA-1 and SHA-256 instructions.
- **Benefit:** Provides very high-throughput hashing of large data. Useful for content-addressed storage of large immutable artifacts such as model checkpoints.

### AES-NI and PCLMULQDQ

- **Path:** `std.Target.x86.Feature.aesni`, `std.Target.x86.Feature.pclmulqdq`
- **What it does:** AES round-function instructions and carry-less multiplication.
- **Benefit:** Accelerates encryption and CRC computation. Relevant if model files or activations need to be authenticated or compressed at high speed.

### CRC32

- **Path:** `std.Target.x86.Feature.crc32`
- **What it does:** Single-cycle CRC32C computation on 8/16/32/64-bit operands.
- **Benefit:** Very fast checksumming of structured data. Frequently used in serialization formats.

---

## Tier 4 — AArch64 equivalents

The following features apply when the target architecture is `aarch64`. They cover Graviton, Apple Silicon, Ampere Altra, and Fujitsu A64FX. They mirror the spirit of the x86 list above.

### NEON

- **Path:** `std.Target.aarch64.Feature.neon`
- **What it does:** Baseline 128-bit SIMD on AArch64.
- **Benefit:** Provides vector arithmetic across all AArch64 implementations.

### FP16

- **Path:** `std.Target.aarch64.Feature.fp16`
- **What it does:** Scalar half-precision floating-point support.
- **Benefit:** Allows efficient scalar FP16 operations without library emulation.

### Full FP16 (FPHP)

- **Path:** `std.Target.aarch64.Feature.fphp`
- **What it does:** Adds FP16 conversion and arithmetic in the SIMD/FP unit, including the inner product instructions.
- **Benefit:** Accelerates FP16 GEMMs and attention kernels.

### Dot product

- **Path:** `std.Target.aarch64.Feature.dotprod`
- **What it does:** Adds `SDOT` and `UDOT`, performing INT8 dot products with INT32 accumulation.
- **Benefit:** Provides INT8 GEMM acceleration equivalent in spirit to AVX-VNNI.

### INT8 matrix multiplication (I8MM)

- **Path:** `std.Target.aarch64.Feature.i8mm`
- **What it does:** Adds `SMMLA`, `UMMLA`, `USDOT`, performing INT8 matrix multiplication on 8x4 tiles.
- **Benefit:** Hardware INT8 GEMM primitive that significantly outperforms emulated solutions.

### BF16

- **Path:** `std.Target.aarch64.Feature.bf16`
- **What it does:** Adds `BFDOT`, `BFMMLA`, and BF16 conversion/arithmetic.
- **Benefit:** Native BF16 inner products, both scalar and 16x8 tile form.

### SVE (Scalable Vector Extension)

- **Path:** `std.Target.aarch64.Feature.sve`
- **What it does:** Introduces scalable vector registers whose length is implementation-defined (typically 128 to 2048 bits).
- **Benefit:** Vector-length-agnostic code that automatically scales with hardware. Critical for high-throughput HPC on ARM server parts.

### SVE2

- **Path:** `std.Target.aarch64.Feature.sve2`
- **What it does:** Extends SVE with additional operations including complex arithmetic, polynomial multiplication, and bitwise permutations.
- **Benefit:** Brings more operations into the scalable vector domain.

### SVE BF16

- **Path:** `std.Target.aarch64.Feature.svebf16`
- **What it does:** Adds BF16 support to SVE.
- **Benefit:** Scalable BF16 GEMMs and reductions.

### SVE matrix-multiply extensions

- **Paths:**
  - `std.Target.aarch64.Feature.svei8mm` — INT8 SVE GEMM
  - `std.Target.aarch64.Feature.svef32mm` — single-precision SVE GEMM
  - `std.Target.aarch64.Feature.svef64mm` — double-precision SVE GEMM
- **What it does:** Provides SVE outer-product primitives.
- **Benefit:** Hardware-accelerated SVE GEMMs at varying precisions.

### SVE atomics

- **Path:** `std.Target.aarch64.Feature.atomics`
- **What it does:** Provides SVE atomic memory operations.
- **Benefit:** Enables atomic updates on SVE vector registers.

### SHA256 and SHA512

- **Path:** `std.Target.aarch64.Feature.sha256`, `std.Target.aarch64.Feature.sha512`
- **What it does:** Hardware SHA accelerators.
- **Benefit:** High-throughput hashing for verification of large blobs.

### AES

- **Path:** `std.Target.aarch64.Feature.aes`
- **What it does:** Hardware AES round function.
- **Benefit:** Fast authenticated encryption and decryption.

### CRC

- **Path:** `std.Target.aarch64.Feature.crc`
- **What it does:** CRC32 computation in a single cycle.
- **Benefit:** Fast checksumming of structured data.

---

## Summary

- Tier 0 features are required baseline: `sse4_1`, `sse4_2`, `avx2`, `fma`, `f16c`, `popcnt`, `lzcnt`, `tzcnt`, `bmi1`, `bmi2`.
- Tier 1 features unlock high throughput: the AVX-512 core (`avx512f`, `avx512dq`, `avx512bw`, `avx512vl`, `avx512cd`), `avx512vnni`, `avx512bf16`, `avx512fp16`, the AMX tile family (`amx_tile`, `amx_int8`, `amx_bf16`), `avx_vnni`, `gfni`, the cache-control primitives (`clflushopt`, `clwb`, `movntdqa`, `movntps`, `movntpd`, `movntdq`), and the platform features `rdrand`, `rdseed`, `fsgsbase`, the `xsave` family, `sha`, `aesni`, `pclmulqdq`, `crc32`.
- Tier 4 features are the AArch64 equivalents: `neon`, `fp16`, `fphp`, `dotprod`, `i8mm`, `bf16`, `sve`, `sve2`, `svebf16`, `svei8mm`, `svef32mm`, `svef64mm`, `atomics`, `sha256`, `sha512`, `aes`, `crc`.

The verification pattern is identical across all tiers and across architectures:

```zig
const enabled = target.cpu.features.isEnabled(@intFromEnum(std.Target.<arch>.Feature.<name>));
```

Replace `<arch>` with `x86`, `x86_64`, or `aarch64` as appropriate, and `<name>` with the path-suffix given above.
