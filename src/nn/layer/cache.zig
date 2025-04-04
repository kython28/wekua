const std = @import("std");
const layer = @import("main.zig");

pub fn Cache(comptime T: type) type {
    const CacheLayer = layer.Layer(T);

    const CacheSlot = struct {
        cache: *anyopaque,
        layer: *const CacheLayer,
    };

    return struct {
        allocator: std.mem.Allocator,
        slots: []CacheSlot,

        const Self = @This();

        pub fn init(
            allocator: std.mem.Allocator,
            number_of_elements: u64,
            layers: []const *const CacheLayer,
        ) !Self {
            const slots = try allocator.alloc(CacheSlot, layers.len);
            var slots_created: usize = 0;
            errdefer {
                for (slots[0..slots_created]) |c| {
                    c.layer.releaseCache(c.cache);
                }
                allocator.free(slots);
            }

            for (layers, slots) |l, *slot| {
                slot.* = .{
                    .cache = try l.prepareCache(number_of_elements),
                    .layer = l,
                };
                slots_created += 1;
            }

            return Self{
                .allocator = allocator,
                .slots = slots,
            };
        }

        pub fn deinit(self: *Self) void {
            const allocator = self.allocator;
            for (self.slots) |c| {
                c.layer.releaseCache(c.cache);
            }
            allocator.free(self.slots);
        }
    };
}
