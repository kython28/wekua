pub const ContentType = enum(u8) {
    tensor = 0x01,
};

pub const WekuaFileHeader = extern struct {
    magic_high: u8 = 0x89,
    magic_w: u8 = 'W',
    content_type: ContentType = .tensor,
    version: u8 = 0x00,
};

pub const TensorMetadata = extern struct {
    type_index: u8,
    ndim: u8,
    reserved: u16 = 0,
};

pub const ValidationError = error{
    InvalidMagic,
    InvalidContentType,
    UnsupportedVersion,
};

pub fn validate(header: WekuaFileHeader, expected_content_type: ContentType) ValidationError!void {
    if (header.magic_high != 0x89 or header.magic_w != 'W') {
        return ValidationError.InvalidMagic;
    }

    if (header.content_type != expected_content_type) {
        return ValidationError.InvalidContentType;
    }

    // NOTE: future versions should use a switch here for backwards-compatible parsing
    if (header.version != 0x00) {
        return ValidationError.UnsupportedVersion;
    }
}
