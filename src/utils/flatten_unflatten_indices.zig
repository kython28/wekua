pub fn ravelMultiIndex(
    multi_index: []const u64,
    shape: []const u64,
    pitches: ?[]const u64,
    is_complex: bool,
) usize {
    if (pitches) |p| {
        var index: u64 = 0;
        for (multi_index, p) |a, b| {
            index += a * b;
        }
        return index;
    }

    var number_of_elements: u64 = 1 + @as(u64, @intFromBool(is_complex));
    for (shape) |e| number_of_elements *= e;

    var index: u64 = 0;
    for (multi_index, shape) |a, b| {
        number_of_elements /= b;
        index += a * number_of_elements;
    }

    return index;
}

pub fn unravelIndex(
    index: u64,
    shape: []const u64,
    pitches: ?[]const u64,
    multi_index: []u64,
    is_complex: bool,
) void {
    if (pitches) |p| {
        var remaining: u64 = index;
        for (p, multi_index) |a, *b| {
            const r = @mod(remaining, a);
            b.* = (remaining - r) / a;
            remaining = r;
        }
        return;
    }

    var number_of_elements: u64 = 1 + @as(u64, @intFromBool(is_complex));
    for (shape) |e| number_of_elements *= e;

    var remaining: u64 = index;
    for (multi_index, shape) |*a, b| {
        number_of_elements /= b;
        const r = @mod(remaining, number_of_elements);
        a.* = (remaining - r) / number_of_elements;
        remaining = r;
    }
}
