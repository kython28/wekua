
pub fn Complex(comptime T: type) type {
    return packed struct {
        real: T,
        imag: T,
    };
}

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

    Complex(i8),
    Complex(u8),
    Complex(i16),
    Complex(u16),
    Complex(i32),
    Complex(u32),
    Complex(i64),
    Complex(u64),
    Complex(f32),
    Complex(f64),
};

pub fn getTypeId(comptime T: type) comptime_int {
    @setEvalBranchQuota(2000);
    inline for (SupportedTypes, 0..) |t, i| {
        if (T == t) {
            return i % 10;
        }
    }

    @compileError("Type not supported");
}

pub fn isComplex(comptime T: type) bool {
    return getTypeId(T) >= 10;
}
