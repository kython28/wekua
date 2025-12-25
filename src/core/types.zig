
fn Complex(comptime T: type) type {
    return packed struct {
        real: T,
        imag: T,
    };
}

pub const ComplexI8 = Complex(i8);
pub const ComplexU8 = Complex(u8);
pub const ComplexI16 = Complex(i16);
pub const ComplexU16 = Complex(u16);
pub const ComplexI32 = Complex(i32);
pub const ComplexU32 = Complex(u32);
pub const ComplexI64 = Complex(i64);
pub const ComplexU64 = Complex(u64);
pub const ComplexF32 = Complex(f32);
pub const ComplexF64 = Complex(f64);

pub const SupportedTypes: [20]type = .{
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    f32,
    f64,

    ComplexI8,
    ComplexU8,
    ComplexI16,
    ComplexU16,
    ComplexI32,
    ComplexU32,
    ComplexI64,
    ComplexU64,
    ComplexF32,
    ComplexF64,
};

pub fn getTypeIndex(comptime T: type) comptime_int {
    // NOTE: This is for avoiding @setEvalBranchQuota
    return switch (T) {
        i8 => 0,
        u8 => 1,
        i16 => 2,
        u16 => 3,
        i32 => 4,
        u32 => 5,
        i64 => 6,
        u64 => 7,
        f32 => 8,
        f64 => 9,

        ComplexI8 => 10,
        ComplexU8 => 11,
        ComplexI16 => 12,
        ComplexU16 => 13,
        ComplexI32 => 14,
        ComplexU32 => 15,
        ComplexI64 => 16,
        ComplexU64 => 17,
        ComplexF32 => 18,
        ComplexF64 => 19,

        else => @compileError("Type not supported"),
    };
}

pub fn getTypeId(comptime T: type) comptime_int {
    // NOTE: This is for avoiding @setEvalBranchQuota
    return switch (T) {
        i8, ComplexI8 => 0,
        u8, ComplexU8 => 1,
        i16, ComplexI16 => 2,
        u16, ComplexU16 => 3,
        i32, ComplexI32 => 4,
        u32, ComplexU32 => 5,
        i64, ComplexI64 => 6,
        u64, ComplexU64 => 7,
        f32, ComplexF32 => 8,
        f64, ComplexF64 => 9,
        else => @compileError("Type not supported"),
    };
}

pub fn isComplex(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct") return false;

    return @hasField(T, "real") and @hasField(T, "imag");
}

pub fn getType(comptime T: type) type {
    return SupportedTypes[getTypeId(T)];
}
