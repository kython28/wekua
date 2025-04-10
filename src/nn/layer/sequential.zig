const wekua = @import("../../wekua.zig");
const std = @import("std");

const w_layer = @import("main.zig");

pub fn Sequential(comptime T: type) type {
    const Tensor = wekua.Tensor(T);
    const Layer = w_layer.Layer(T);

    const SequentialCache = struct {
        caches: []*anyopaque,
        gradients: []*Tensor,
        bias_gradients: []?*Tensor,
    };

    return struct {
        pub const Cache = Layer.Cache;

        allocator: std.mem.Allocator,
        layers: std.ArrayList(Layer),
        weights: std.ArrayList(*Tensor),
        bias: std.ArrayList(?*Tensor),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const seq_layer = try allocator.create(Self);
            errdefer allocator.destroy(seq_layer);

            seq_layer.* = .{
                .allocator = allocator,
                .layers = std.ArrayList(Layer).init(allocator),
                .weights = std.ArrayList(*Tensor).init(allocator),
                .bias = std.ArrayList(?*Tensor).init(allocator),
            };

            return seq_layer;
        }

        pub fn layer(self: *Self) Layer {
            return Layer{
                .vtable = .{
                    .deinit = &layer_deinit,
                    .getCachedOutput = &getCachedOutput,
                    .getWeights = &getWeights,
                    .getBias = &getBias,
                    .prepareCache = &prepareCache,
                    .releaseCache = &releaseCache,
                    .forward = &forward,
                    .getSensitivity = &getSensitivity,
                    .backward = &backward,
                    .getGradients = &getGradients,
                    .getBiasGradients = &getBiasGradients,
                },
                .ptr = self,
            };
        }

        fn layer_deinit(ptr: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            self.deinit();
        }

        pub fn deinit(self: *const Self) void {
            for (self.layers.items) |l| {
                l.deinit();
            }
            self.layers.deinit();
            self.weights.deinit();
            self.bias.deinit();

            self.allocator.destroy(self);
        }

        pub fn append(self: *Self, new_layer: Layer) !void {
            try self.layers.append(new_layer);
            errdefer _ = self.layers.pop();

            const weights = new_layer.getWeights();
            const bias = new_layer.getBias();

            try self.weights.appendSlice(weights);
            errdefer {
                for (0..weights.len) |_| {
                    _ = self.weights.pop();
                }
            }

            if (bias) |b| {
                try self.bias.appendSlice(b);
            }else{
                try self.bias.ensureTotalCapacity(self.bias.capacity + weights.len);
                for (weights) |_| {
                    self.bias.appendAssumeCapacity(null);
                }
            }
            errdefer {
                for (0..weights.len) |_| {
                    _ = self.bias.pop();
                }
            }
        }

        fn getCachedOutput(ptr: *const anyopaque, cache: *const anyopaque) *Tensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            const last_index = self.layers.items.len - 1;

            const last_layer_cache = cache_data.caches[last_index];
            return self.layers.items[last_index].getCachedOutput(last_layer_cache);
        }

        fn getWeights(ptr: *const anyopaque) []const *Tensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            return self.weights.items;
        }

        fn getBias(ptr: *const anyopaque) ?[]const ?*Tensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            
            return self.bias.items;
        }

        fn prepareCache(
            ptr: *const anyopaque,
            number_of_elements: u64,
        ) !*anyopaque {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            const caches = try allocator.alloc(*anyopaque, self.layers.items.len);
            errdefer allocator.free(caches);

            const gradients = try allocator.alloc(*Tensor, self.layers.items.len);
            errdefer allocator.free(gradients);

            const bias_gradients = try allocator.alloc(?*Tensor, self.layers.items.len);
            errdefer allocator.free(bias_gradients);

            var caches_created: usize = 0;
            errdefer {
                for (self.layers.items, caches[0..caches_created]) |l, c| {
                    l.releaseCache(c);
                }
                allocator.free(caches);
            }

            var items: usize = 0;
            for (self.layers.items, caches) |l, *c| {
                c.* = try l.prepareCache(number_of_elements);
                errdefer l.releaseCache(c.*);

                items += l.getGradients(c.*).len;
                caches_created += 1;
            }

            var offset: usize = 0;
            for (self.layers.items, 0..) |l, index| {
                const l_gradients = l.getGradients(caches[index]);

                const new_offset = offset + l_gradients.len;
                @memcpy(gradients[offset..new_offset], l_gradients);

                if (l.getBiasGradients(caches[index])) |l_bias_gradients| {
                    @memcpy(bias_gradients[offset..new_offset], l_bias_gradients);
                }else{
                    @memset(bias_gradients[offset..new_offset], null);
                }

                offset = new_offset;
            }

            const cache = try allocator.create(SequentialCache);
            errdefer allocator.destroy(cache);

            cache.* = .{
                .caches = caches,
                .gradients = gradients,
                .bias_gradients = bias_gradients,
            };

            return cache;
        }

        fn releaseCache(ptr: *const anyopaque, cache: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));

            for (self.layers.items, cache_data.caches) |l, c| {
                l.releaseCache(c);
            }

            allocator.free(cache_data.caches);
            allocator.free(cache_data.gradients);
            allocator.free(cache_data.bias_gradients);
            allocator.destroy(cache_data);
        }

        fn forward(
            ptr: *const anyopaque,
            command_queue: *const wekua.core.CommandQueue,
            input: *Tensor,
            cache: *anyopaque,
        ) !*Tensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            const layers = self.layers.items;

            var output = input;
            for (layers, cache_data.caches) |l, c| {
                output = try l.forward(command_queue, output, c);
            }

            return output;
        }

        fn getSensitivity(ptr: *const anyopaque, cache: *const anyopaque) *Tensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            const layers = self.layers.items;

            const last_layer: *const Layer = &layers[layers.len - 1];
            const sensitivity = last_layer.getSensitivity(cache_data.caches[layers.len - 1]);

            return sensitivity;
        }

        fn backward(
            ptr: *const anyopaque,
            command_queue: *const wekua.core.CommandQueue,
            cache: *anyopaque,
            input: *Tensor,
            input_gradient: ?*Tensor,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            const caches = cache_data.caches;
            const layers = self.layers.items;

            var index: usize = layers.len - 1;
            while (true) {
                const _input = blk: {
                    if (index == 0) break :blk input;

                    break :blk layers[index - 1].getCachedOutput(caches[index - 1]);
                };

                const _input_gradient = blk: {
                    if (index == 0) break :blk input_gradient;

                    break :blk layers[index - 1].getSensitivity(caches[index - 1]);
                };

                try layers[index].backward(command_queue, caches[index], _input, _input_gradient);

                if (index == 0) break;
                index -= 1;
            }
        }

        fn getGradients(_: *const anyopaque, cache: *const anyopaque) []const *Tensor {
            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            return cache_data.gradients;
        }

        fn getBiasGradients(_: *const anyopaque, cache: *const anyopaque) ?[]const ?*Tensor {
            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            return cache_data.bias_gradients;
        }
    };
}
