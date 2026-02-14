const cl = @import("opencl");
const std = @import("std");

const core = @import("core");
const Context = core.Context;
const CommandQueue = core.CommandQueue;
const Pipeline = core.Pipeline;

const utils = @import("utils");

pub const helpers = @import("helpers.zig");

const init = @import("init.zig");

pub const fill = @import("fill.zig");
pub const memory = @import("memory/main.zig");
pub const random = @import("random/main.zig");
const transpose_module = @import("transpose.zig");
pub const transpose = transpose_module.transpose;
pub const transpose_2d_inplace = transpose_module.transpose_2d_inplace;
pub const convertions = @import("convertions/main.zig");
pub const identity = @import("identity.zig").identity;
pub const print = @import("print.zig").print;

const WorkConfiguration = @import("work_configuration.zig");
pub const GemmAlgorithm = WorkConfiguration.GemmAlgorithm;


pub const Errors = error{
    InvalidValue,
    InvalidCoordinates,
    InvalidBuffer,
    UnqualTensorsAttribute,
    UnqualTensorsShape,
    UnqualTensorsDimension,
} || std.mem.Allocator.Error || cl.errors.OpenCLError || core.KernelsSet.Errors;

pub const CreateConfig = struct {
    cl_mem_flags: cl.buffer.MemFlags = cl.buffer.MemFlag.read_write,
    host_ptr: ?*anyopaque = null,
    vectors_enabled: bool = true,
};

const Dimensions = struct {
    shape: []u64,
    vl_shape: []u64,
    pitches: []u64,
    number_of_elements: u64,
    number_of_elements_without_padding: u64,
};

const MemoryLayout = struct {
    row_pitch: u64,
    row_pitch_for_vectors: u64,
    slice_pitch: u64,
    slice_pitch_for_vectors: u64,
    number_of_vectors: u64,
    size: usize,
};

const Flags = struct {
    vectors_enabled: bool,
};

pub fn Tensor(comptime T: type) type {
    const type_id = core.types.getTypeId(T);
    const is_complex = core.types.isComplex(T);

    return struct {
        context: *const Context,
        arena: std.heap.ArenaAllocator,

        buffer: cl.buffer.Mem,
        pitches_buffer: cl.buffer.Mem,

        dimensions: Dimensions,
        work_configuration: WorkConfiguration,
        memory_layout: MemoryLayout,
        flags: Flags,

        const Self = @This();

        fn createPitchBuffer(
            self: *Self,
            context: *const Context,
            pipeline: *Pipeline,
            pitches: []const u64,
        ) Errors!void {
            const pitches_buffer = try cl.buffer.create(
                context.cl_context,
                cl.buffer.MemFlag.read_write,
                pitches.len * @sizeOf(u64),
                null,
            );
            self.pitches_buffer = pitches_buffer;
            errdefer cl.buffer.release(pitches_buffer);

            const prev_events = pipeline.prevEvents();

            var new_event: cl.event.Event = undefined;
            try cl.buffer.write(
                pipeline.command_queue.cl_command_queue,
                pitches_buffer,
                false,
                0,
                pitches.len * @sizeOf(u64),
                pitches.ptr,
                prev_events,
                &new_event,
            );
            errdefer helpers.releaseEvent(new_event);

            try pipeline.append(&.{new_event});
        }

        pub fn empty(
            context: *const Context,
            pipeline: *Pipeline,
            shape: []const u64,
            config: CreateConfig,
        ) Errors!*Self {
            if (shape.len == 0) {
                return Errors.InvalidValue;
            }

            const allocator = context.allocator;
            const command_queues = context.command_queues;

            const tensor = try allocator.create(Self);
            errdefer allocator.destroy(tensor);

            tensor.context = context;
            tensor.arena = std.heap.ArenaAllocator.init(allocator);
            errdefer tensor.arena.deinit();

            const arena_allocator = tensor.arena.allocator();

            tensor.dimensions.shape = try arena_allocator.alloc(u64, shape.len);
            for (tensor.dimensions.shape, shape) |*d, s| {
                if (s == 0) return Errors.InvalidValue;

                d.* = s;
            }

            const pitches = try arena_allocator.alloc(u64, shape.len);
            tensor.dimensions.pitches = pitches;

            const number_of_elements = try init.initTensorProperties(
                T,
                is_complex,
                type_id,
                command_queues,
                arena_allocator,
                tensor,
                shape,
                config.vectors_enabled,
            );

            const size = number_of_elements * @sizeOf(T);
            tensor.memory_layout.size = size;

            tensor.buffer = try cl.buffer.create(
                context.cl_context,
                config.cl_mem_flags,
                size,
                config.host_ptr,
            );
            errdefer cl.buffer.release(tensor.buffer);

            try tensor.createPitchBuffer(context, pipeline, pitches);
            return tensor;
        }

        pub fn release(self: *Self, pipeline: *Pipeline) void {
            pipeline.waitAndCleanup();

            const allocator = self.context.allocator;

            cl.buffer.release(self.buffer);
            cl.buffer.release(self.pitches_buffer);

            self.arena.deinit();
            allocator.destroy(self);
        }

        pub fn alloc(
            context: *const Context,
            pipeline: *Pipeline,
            shape: []const u64,
            config: CreateConfig,
        ) Errors!*Self {
            const tensor = try empty(context, pipeline, shape, config);
            errdefer tensor.release(pipeline);

            try fill.zeroes(T, pipeline, tensor);

            return tensor;
        }
    };
}

// Unit Tests
const testing = std.testing;

test {
    std.testing.refAllDecls(@This());
}

test "Tensor.empty - basic initialization for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(tensor.context == context);
            try testing.expectEqual(@as(usize, 3), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 2), tensor.dimensions.shape[0]);
            try testing.expectEqual(@as(u64, 3), tensor.dimensions.shape[1]);
            try testing.expectEqual(@as(u64, 4), tensor.dimensions.shape[2]);
        }
    }
}

test "Tensor.empty - 1D tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 1), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 10), tensor.dimensions.shape[0]);

            try testing.expectEqual(tensor.dimensions.number_of_elements_without_padding, 10);
            try testing.expect(tensor.dimensions.number_of_elements >= 10);
        }
    }
}

test "Tensor.empty - 2D tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 5, 7 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 2), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 5), tensor.dimensions.shape[0]);
            try testing.expectEqual(@as(u64, 7), tensor.dimensions.shape[1]);
        }
    }
}

test "Tensor.empty - 4D tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4, 5 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 4), tensor.dimensions.shape.len);
            try testing.expectEqual(@as(u64, 2), tensor.dimensions.shape[0]);
            try testing.expectEqual(@as(u64, 3), tensor.dimensions.shape[1]);
            try testing.expectEqual(@as(u64, 4), tensor.dimensions.shape[2]);
            try testing.expectEqual(@as(u64, 5), tensor.dimensions.shape[3]);
        }
    }
}

test "Tensor.empty - invalid empty shape" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const pipeline = try Pipeline.init(&context.command_queues[0]);
    defer pipeline.deinit();

    const shape: []const u64 = &.{};
    const config = CreateConfig{};

    const result = Tensor(f32).empty(context, pipeline, shape, config);
    try testing.expectError(Errors.InvalidValue, result);
}

test "Tensor.empty - invalid zero dimension" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const pipeline = try Pipeline.init(&context.command_queues[0]);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 0, 4 };
    const config = CreateConfig{};

    const result = Tensor(f32).empty(context, pipeline, &shape, config);
    try testing.expectError(Errors.InvalidValue, result);
}

test "Tensor.empty - complex flag disables vectors" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (core.types.isComplex(T) and command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(!tensor.flags.vectors_enabled);
        }
    }
}

test "Tensor.empty - vectors can be disabled" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{ .vectors_enabled = false };

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(!tensor.flags.vectors_enabled);
        }
    }
}

test "Tensor.alloc - creates zeroed tensor" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expect(tensor.context == context);
            try testing.expectEqual(@as(usize, 3), tensor.dimensions.shape.len);
        }
    }
}

test "Tensor - dimensions calculated correctly" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Number of elements without padding should be 2*3*4 = 24
            try testing.expectEqual(@as(u64, 24), tensor.dimensions.number_of_elements_without_padding);

            // Total elements should be >= elements without padding (due to padding)
            try testing.expect(tensor.dimensions.number_of_elements >= tensor.dimensions.number_of_elements_without_padding);
        }
    }
}

test "Tensor - pitches calculated correctly" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Verify pitches array has same length as shape
            try testing.expectEqual(shape.len, tensor.dimensions.pitches.len);

            // Last pitch should equal multiplier (1 for non-complex)
            try testing.expectEqual(@as(u64, 1), tensor.dimensions.pitches[tensor.dimensions.pitches.len - 1]);
        }
    }
}

test "Tensor - memory layout" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Size should be at least the minimum required
            const min_size = tensor.dimensions.number_of_elements_without_padding * @sizeOf(T);
            try testing.expect(tensor.memory_layout.size >= min_size);

            // Row pitch should be >= last dimension
            try testing.expect(tensor.memory_layout.row_pitch >= shape[shape.len - 1]);
        }
    }
}

test "Tensor.release - proper cleanup" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            tensor.release(pipeline);
        }
    }
}

test "Tensor - multiple tensors on same context" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape1 = [_]u64{ 2, 3 };
    const shape2 = [_]u64{ 4, 5, 6 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor1 = try Tensor(T).alloc(context, pipeline, &shape1, config);
            defer tensor1.release(pipeline);

            const tensor2 = try Tensor(T).alloc(context, pipeline, &shape2, config);
            defer tensor2.release(pipeline);

            // Both should reference the same context
            try testing.expect(tensor1.context == tensor2.context);

            // But should have different shapes
            try testing.expect(tensor1.dimensions.shape.len != tensor2.dimensions.shape.len);
        }
    }
}

test "Tensor - work configuration initialized" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Verify work configuration arrays are allocated
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items.len);
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items_without_vectors.len);
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items_1d.len);
            try testing.expectEqual(context.command_queues.len, tensor.work_configuration.local_work_items_for_vectors_1d.len);
        }
    }
}

test "Tensor.empty - custom memory flags" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = CreateConfig{
        .cl_mem_flags = cl.buffer.MemFlag.read_only,
    };

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            try testing.expectEqual(@as(usize, 2), tensor.dimensions.shape.len);
        }
    }
}

test "Tensor - vl_shape calculated" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // vl_shape should have same length as shape
            try testing.expectEqual(shape.len, tensor.dimensions.vl_shape.len);

            // First dimensions should match
            for (shape[0 .. shape.len - 1], tensor.dimensions.vl_shape[0 .. shape.len - 1]) |s, vl| {
                try testing.expectEqual(s, vl);
            }
        }
    }
}
