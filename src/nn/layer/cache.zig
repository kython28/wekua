const std = @import("std");
const layer = @import("main.zig");

const core = @import("core");
const Pipeline = core.Pipeline;

const tensor_module = @import("tensor");

pub fn Cache(comptime T: type) type {
    const CacheLayer = layer.Layer(T);

    const CacheSlot = struct {
        cache: *anyopaque,
        layer: *const CacheLayer,
    };

    return struct {
        allocator: std.mem.Allocator,
        slots: []CacheSlot,

        error_tensor: *tensor_module.Tensor(T),

        const Self = @This();

        pub fn init(
            context: *const core.Context,
            pipeline: *Pipeline,
            number_of_elements: u64,
            layers: []const *const CacheLayer,
        ) !Self {
            const allocator = context.allocator;
            const slots = try allocator.alloc(CacheSlot, layers.len);
            var slots_created: usize = 0;
            errdefer {
                for (slots[0..slots_created]) |c| {
                    c.layer.releaseCache(pipeline, c.cache);
                }
                allocator.free(slots);
            }

            for (layers, slots) |l, *slot| {
                slot.* = .{
                    .cache = try l.prepareCache(pipeline, number_of_elements),
                    .layer = l,
                };
                slots_created += 1;
            }

            const last_slot = slots[slots_created - 1];
            const last_sensitivity = last_slot.layer.getSensitivity(last_slot.cache);

            const error_tensor = try tensor_module.Tensor(T).alloc(context, pipeline, last_sensitivity.dimensions.shape, .{});
            errdefer error_tensor.release(pipeline);

            return Self{
                .allocator = allocator,
                .slots = slots,
                .error_tensor = error_tensor,
            };
        }

        pub inline fn getLayerCache(self: *const Self, index: usize) *anyopaque {
            return self.slots[index].cache;
        }

        pub fn deinit(self: *const Self, pipeline: *Pipeline) void {
            const allocator = self.allocator;
            for (self.slots) |c| {
                c.layer.releaseCache(pipeline, c.cache);
            }
            allocator.free(self.slots);
            self.error_tensor.release(pipeline);
        }
    };
}
