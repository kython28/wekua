# Inline Assembly in Zig

Zig exposes inline assembly as a first-class expression. The expression emits the requested assembly exactly as written, integrates with Zig's type system, and uses an operand-binding syntax that scales from trivial register moves to complex data routing.

This document covers the full syntax of the `asm` expression, the operand-binding language, AT&T syntax on x86, and several worked examples.

---

## General form

```zig
asm volatile (
    \\<assembly text>
    : <outputs>
    : <inputs>
    : <clobbers>
);
```

The four sections are separated by colons. All four sections are syntactically required, although the outputs and inputs may be empty (the colon delimiters remain). The `volatile` keyword is optional but strongly recommended whenever the assembly has side effects.

The expression yields a value whose type is the type given at the end of the outputs section. If the outputs section ends with a type rather than a value binding, the assembly must place its result into the register specified by the output constraint.

---

## The `volatile` modifier

```zig
asm volatile (...);
```

Without `volatile`, Zig is permitted to delete the inline assembly entirely if the resulting value is unused. With `volatile`, the compiler treats the expression as having observable side effects and will never elide it. Always use `volatile` when the assembly performs I/O, memory writes, control flow, or anything else that must execute.

---

## The assembly string

The assembly text is a comptime string. Multi-line raw string literals using `\\` are idiomatic because they preserve the assembly formatting and avoid having to escape newlines.

Inside the assembly string, `%` has special meaning. To emit a literal `%`, write `%%`. The text between percent signs refers to one of three things:

1. A register substitution written as `%[name]`, where `name` matches the binding name of an output or input.
2. A numbered operand `%0`, `%1`, `%2`, … in left-to-right order of the outputs and inputs combined.
3. The literal `%` itself (when doubled to `%%`).

The AT&T dialect uses `$` to mark register names, so a typical substitution looks like `%[number]`, which expands to the input operand bound to the name `number`.

---

## Outputs section

The outputs section follows the assembly string. Each output is a tuple of four parts:

1. **Binding name** — used by `%[name]` in the assembly string. Optional.
2. **Constraint string** — describes where the result is placed. Common forms:
   - `"={rax}"` — place the result into `%rax`, clobbering its previous value.
   - `"=r"` — let the compiler pick any general-purpose register.
   - `"=m"` — write the result directly to memory at the address of an input pointer.
   - `"={rax},={rdx}"` — multiple outputs are not currently supported in Zig's inline asm; use registers that are read back in subsequent code.
3. **Result type or value binding** — if a type is given, the assembly expression yields that type. If a value binding is given (using the `[name] (-> Type)` form), the output register is tied to a Zig value that can be referenced by name elsewhere.
4. **Variadic flags** — for inputs only.

The full grammar for a single output entry is:

```zig
[<name>] <constraint> (-> <Type>)
```

or

```zig
[<name>] <constraint> (<value>)
```

The value-binding form is rare. Most code uses the type form because the assembly expression's value is read directly.

### Example: single output, type form

```zig
const result: u64 = asm volatile (
    \\rdtsc
    : [ret] "={rax},={rdx}" (-> u64),
);
```

This reads the time-stamp counter. Because `rdtsc` writes both `%rax` and `%rdx`, the constraint `"={rax},={rdx}"` declares both outputs.

### Example: no outputs

```zig
asm volatile ("nop" : : : );
```

The two adjacent colons mark the empty outputs section.

---

## Inputs section

The inputs section follows the outputs. Each input entry follows:

```zig
[<name>] <constraint> (<value>)
```

Where `<value>` is a Zig expression evaluated at the call site. The constraint describes how the value is delivered. Common forms:

- `"r"` — pick any general-purpose register.
- `"{rax}"` — force the value into `%rax`.
- `"i"` — immediate value, baked into the instruction stream.
- `"m"` — pass through a memory location.
- `"rm"` — pick register or memory.

Numeric constraint references:

- `"{rax}"` is equivalent to `"0"` when applied to an input that should occupy the same register as output `0`.
- A constraint like `"r"` paired with a leading `=` makes the operand read-write.

### Example: immediate operand

```zig
asm volatile (
    \\movb %[val], %%al
    : [ret] "={al}" (-> u8),
    : [val] "i" (@as(u8, 7)),
);
```

The `@as(u8, 7)` ensures the comptime integer fits the operand type.

---

## Clobbers section

The clobbers section is a struct literal that declares registers the assembly modifies without using the input/output mechanism. The struct fields are the register names without the `%` prefix.

```zig
.{ .rcx = true, .r11 = true }
```

The special clobber `.memory = true` declares that the assembly writes to arbitrary undeclared memory. This blocks the compiler from reordering loads and stores around the assembly.

```zig
.{ .memory = true, .cc = true }
```

The `.cc = true` clobber declares that the assembly modifies the condition codes, preventing the compiler from making assumptions about flag state across the assembly.

Common x86-64 clobbers:

- `.rax`, `.rbx`, `.rcx`, `.rdx`, `.rsi`, `.rdi`, `.rsp`, `.rbp`
- `.r8` … `.r15`
- `.xmm0` … `.xmm15`
- `.ymm0` … `.ymm15`
- `.zmm0` … `.zmm31`
- `.memory`
- `.cc`

On AArch64, clobbers use `.x0` … `.x30` and `.v0` … `.v31`.

---

## Worked example: syscall

```zig
pub fn syscall1(number: usize, arg1: usize) usize {
    return asm volatile (
        \\syscall
        : [ret] "={rax}" (-> usize),
        : [number] "{rax}" (number),
          [arg1]   "{rdi}" (arg1),
        : .{ .rcx = true, .r11 = true });
}
```

Notes:

- The `syscall` instruction places the return value in `%rax` and clobbers `%rcx` and `%r11`. The kernel also writes `%rcx` (with the return address it had to save) and `%r11` (with the saved RFLAGS), which is why those are listed.
- The output constraint `"={rax}"` ties the value of `%rax` to the expression's result type `usize`.
- The first input `"${rax}"` puts `number` into `%rax`, which is the convention for the syscall number.
- The second input `"${rdi}"` puts `arg1` into `%rdi`, the first argument register in the System V AMD64 ABI.

---

## Worked example: load FPU control word

```zig
pub fn readMxcsr() u32 {
    return asm volatile (
        \\stmxcsr %[ret]
        : [ret] "=m" (-> u32),
    );
}
```

The constraint `"=m"` writes the result directly into a stack slot that Zig then reads as a `u32`. This is useful when the destination is a memory operand rather than a register.

---

## Worked example: load via `cpuid`

```zig
pub fn cpuid(leaf: u32) struct { eax: u32, ebx: u32, ecx: u32, edx: u32 } {
    return asm volatile (
        \\cpuid
        : [eax] "={eax}" (-> u32),
          [ebx] "={ebx}" (-> u32),
          [ecx] "={ecx}" (-> u32),
          [edx] "={edx}" (-> u32),
        : [leaf] "{eax}" (leaf),
        : .{ .rbx = true });
}
```

Zig currently exposes only a single output slot per `asm` expression, so when several registers need to be returned, the recommended pattern is to wrap the assembly in a function that uses tuple structs or anonymous structs with multiple outputs. The example above uses Zig's anonymous struct return value and binds each register to its own field; note that Zig supports multiple outputs as a tuple.

---

## AT&T syntax on x86 and x86_64

For x86 and x86_64 targets, Zig accepts only AT&T syntax. This is a deliberate choice: assembly parsing is delegated to LLVM, and LLVM's support for Intel syntax is unreliable and undertested. AT&T syntax has these distinguishing features:

- **Register prefix:** `%` before each register name. Inside the assembly string, `%[name]` still denotes a Zig operand, so to refer to `%rax` literally you write `%%rax`. The double percent escapes to a single literal `%`.
- **Operand order:** source first, destination second. `movq %rax, %rbx` moves `%rax` into `%rbx`.
- **Operand size suffixes:** `b` (8-bit), `w` (16-bit), `l` (32-bit), `q` (64-bit), `s` (single-precision float), `d` (double-precision float). Example: `addq $1, %rax`, `movss %xmm0, %xmm1`.
- **Immediate prefix:** `$` before immediates. `movq $42, %rax` loads the constant 42.
- **Memory operands:** `disp(%base, %index, scale)`. Example: `movq 8(%rsp), %rax` loads a quad from `[%rsp + 8]`. Example with index: `movq (%rdi,%rcx,8), %rax` loads from `[%rdi + %rcx*8]`.
- **Indirect through a label:** use `$` with parentheses. Example: `leaq _my_var(%rip), %rax` produces a position-independent address.

A quick conversion table:

| Concept        | Intel syntax         | AT&T syntax                  |
|----------------|----------------------|------------------------------|
| Move rax → rbx | `mov rbx, rax`       | `movq %rax, %rbx`            |
| Add 1 to rax   | `add rax, 1`         | `addq $1, %rax`              |
| Load constant  | `mov rax, 0x10`      | `movq $0x10, %rax`           |
| Load memory    | `mov rax, [rsp+8]`   | `movq 8(%rsp), %rax`         |
| Compare        | `cmp rax, rbx`       | `cmpq %rbx, %rax`            |
| Jump label     | `jmp .loop`          | `jmp .loop` (same syntax)    |

Memory operands in constraints are translated by LLVM, so `m` constraints work identically regardless of dialect. The dialect only affects what is written inside the assembly string.

---

## Global labels and PC-relative addressing

Inside an `asm` expression, named labels in the same translation unit are referenced with their bare identifier. Inside an `asm` block embedded in a function, references to a function-local label use the label's name.

For symbols in the same compilation unit, AT&T syntax uses `_symbol(%rip)` to load the address position-independently. Inside the assembly string, this looks like:

```zig
asm volatile (
    \\leaq _my_global(%rip), %[out]
    : [out] "=r" (-> usize),
);
```

---

## Memory clobber and the optimizer

Zig's optimizer will, in the absence of `.memory = true`, assume that the assembly does not touch memory unless it is told so by an input or output. If the assembly reads from or writes to addresses that are not part of an explicit operand, you must declare `.memory = true`. This is required for syscalls, MMIO, and any volatile pointer dereference.

The optimizer also assumes that condition codes are preserved unless `.cc = true` is declared. Most instructions that touch `%eflags` should declare this clobber.

---

## AArch64 differences

AArch64 inline assembly in Zig uses the same `asm` syntax but the assembly text uses standard AArch64 syntax (not AT&T, not Intel). Operand constraints use plain register names without braces.

```zig
pub fn readCntfrqEl0() u64 {
    return asm volatile (
        \\mrs %[ret], cntfrq_el0
        : [ret] "=r" (-> u64),
    );
}
```

Clobbers on AArch64 use the `.x0` … `.x30` and `.v0` … `.v31` form.

---

## Comptime string semantics

The assembly text is comptime-known. This means:

- The text can be built with comptime concatenation and comptime `if` to select different assembly per target.
- Conditional compilation of operands can be done at comptime.

Example: one assembly expression per target.

```zig
const result: u64 = if (builtin.cpu.arch.isX86())
    asm volatile (
        \\rdtsc
        : [ret] "={rax},={rdx}" (-> u64),
    )
else if (builtin.cpu.arch.isAARCH64())
    asm volatile (
        \\mrs %[ret], cntvct_el0
        : [ret] "=r" (-> u64),
    )
else
    @compileError("unsupported architecture");
```

Because the entire expression is comptime-evaluated for its template, only the relevant branch's assembly is ever generated.

---

## Pitfalls and best practices

- Always include `volatile` when the assembly has side effects.
- Always declare every register that the assembly writes to. Zig will not infer it from the assembly text.
- Always include `.memory = true` if the assembly reads or writes memory not represented in operands.
- Always include `.cc = true` if the assembly modifies condition codes.
- Prefer `=r` over explicit register constraints when possible, so Zig can pick registers that avoid spills.
- Use `%[name]` rather than numbered `%0`, `%1`, etc., because operand order can change and the compiler will issue a comptime error if a numbered operand is out of bounds.
- When mixing register-sized operands, the constraint (not the suffix) carries the size; in AT&T syntax the suffix is still required for correctness, so keep both consistent.

---

## Further references

Zig's inline assembly feature is built on LLVM's inline assembler. The constraint-string semantics follow:

- LLVM Language Reference, Inline Assembler Constraint Strings: https://llvm.org/docs/LangRef.html#inline-asm-constraint-string
- GCC Extended Asm documentation: https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html

Both documents describe the meaning of each constraint string. Zig does not reimplement any of that; it forwards the constraint verbatim.
