const std = @import("std");

const core = @import("core");
const Pipeline = core.Pipeline;

const tensor_module = @import("tensor");
const TensorErrors = tensor_module.Errors;

const layer_module = @import("main.zig");

pub fn Sequential(comptime T: type) type {
    const Tensor = tensor_module.Tensor(T);
    const Layer = layer_module.Layer(T);

    const SequentialCache = struct {
        caches: []*anyopaque,
        gradients: []*Tensor,
        bias_gradients: []?*Tensor,
    };

    return struct {
        allocator: std.mem.Allocator,
        layers: std.ArrayList(Layer),
        weights: std.ArrayList(*Tensor),
        bias: std.ArrayList(?*Tensor),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) TensorErrors!*Self {
            const seq_layer = try allocator.create(Self);
            errdefer allocator.destroy(seq_layer);

            seq_layer.* = .{
                .allocator = allocator,
                .layers = .empty,
                .weights = .empty,
                .bias = .empty,
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

        fn layer_deinit(ptr: *anyopaque, pipeline: *Pipeline) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.deinit(pipeline);
        }

        pub fn deinit(self: *Self, pipeline: *Pipeline) void {
            for (self.layers.items) |l| {
                l.deinit(pipeline);
            }
            const allocator = self.allocator;

            self.layers.deinit(allocator);
            self.weights.deinit(allocator);
            self.bias.deinit(allocator);

            allocator.destroy(self);
        }

        pub fn append(self: *Self, new_layer: Layer) TensorErrors!void {
            const allocator = self.allocator;

            try self.layers.append(allocator, new_layer);
            errdefer _ = self.layers.pop();

            const layer_weights = new_layer.getWeights();
            const layer_bias = new_layer.getBias();

            try self.weights.appendSlice(allocator, layer_weights);
            errdefer {
                for (0..layer_weights.len) |_| {
                    _ = self.weights.pop();
                }
            }

            if (layer_bias) |b| {
                try self.bias.appendSlice(allocator, b);
            } else {
                try self.bias.ensureTotalCapacity(allocator, self.bias.capacity + layer_weights.len);
                for (layer_weights) |_| {
                    self.bias.appendAssumeCapacity(null);
                }
            }
            errdefer {
                for (0..layer_weights.len) |_| {
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
            pipeline: *Pipeline,
            number_of_elements: u64,
        ) TensorErrors!*anyopaque {
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
                    l.releaseCache(pipeline, c);
                }
                allocator.free(caches);
            }

            var items: usize = 0;
            for (self.layers.items, caches) |l, *c| {
                c.* = try l.prepareCache(pipeline, number_of_elements);
                errdefer l.releaseCache(pipeline, c.*);

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
                } else {
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

        fn releaseCache(ptr: *const anyopaque, pipeline: *Pipeline, cache: *const anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const allocator = self.allocator;
            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));

            for (self.layers.items, cache_data.caches) |l, c| {
                l.releaseCache(pipeline, c);
            }

            allocator.free(cache_data.caches);
            allocator.free(cache_data.gradients);
            allocator.free(cache_data.bias_gradients);
            allocator.destroy(cache_data);
        }

        fn forward(
            ptr: *const anyopaque,
            pipeline: *Pipeline,
            input_tensor: *Tensor,
            cache: *anyopaque,
        ) TensorErrors!*Tensor {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            const layers = self.layers.items;

            var output = input_tensor;
            for (layers, cache_data.caches) |l, c| {
                output = try l.forward(pipeline, output, c);
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
            pipeline: *Pipeline,
            cache: *anyopaque,
            input_tensor: *Tensor,
            input_gradient: ?*Tensor,
        ) TensorErrors!void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            const cache_data: *const SequentialCache = @ptrCast(@alignCast(cache));
            const caches = cache_data.caches;
            const layers = self.layers.items;

            var index: usize = layers.len - 1;
            while (true) {
                const _input = blk: {
                    if (index == 0) break :blk input_tensor;

                    break :blk layers[index - 1].getCachedOutput(caches[index - 1]);
                };

                const _input_gradient = blk: {
                    if (index == 0) break :blk input_gradient;

                    break :blk layers[index - 1].getSensitivity(caches[index - 1]);
                };

                try layers[index].backward(pipeline, caches[index], _input, _input_gradient);

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
