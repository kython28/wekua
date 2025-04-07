const std = @import("std");
const layer = @import("main.zig");

const wekua = @import("../../wekua.zig");

pub fn Cache(comptime T: type) type {
    const CacheLayer = layer.Layer(T);

    const CacheSlot = struct {
        cache: *anyopaque,
        layer: *const CacheLayer,
    };

    return struct {
        allocator: std.mem.Allocator,
        slots: []CacheSlot,

        error_tensor: *wekua.Tensor(T),

        const Self = @This();

        pub fn init(
            context: *const wekua.core.Context,
            number_of_elements: u64,
            layers: []const *const CacheLayer,
        ) !Self {
            const allocator = context.allocator;
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

            const last_slot = slots[slots_created - 1];
            const last_gradient = last_slot.layer.getGradient(last_slot.cache);

            const error_tensor = try wekua.Tensor(T).alloc(context, last_gradient.dimensions.shape, .{});
            errdefer error_tensor.release();

            return Self{
                .allocator = allocator,
                .slots = slots,
                .error_tensor = error_tensor,
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
