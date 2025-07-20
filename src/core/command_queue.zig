const std = @import("std");
const cl = @import("opencl");

const Context = @import("context.zig");
const KernelsSet = @import("kernel.zig");

allocator: std.mem.Allocator,
ctx: *const Context,
cmd: cl.command_queue.CommandQueue,
device: cl.device.DeviceId,

device_name: []u8,
device_vendor_id: u32,
device_type: cl.device.Type,

kernels: [KernelsSet.TOTAL_NUMBER_OF_KERNELS]KernelsSet,
headers: KernelsSet,

local_mem_type: cl.device.LocalMemType,
local_mem_size: u64,

compute_units: u32,
vector_widths: [10]u32,
max_work_group_size: u64,
cache_line_size: u32,
wekua_id: usize,

fn get_device_info(
    self: *CommandQueue,
    allocator: std.mem.Allocator,
    device: cl.device.DeviceId,
) !void {
    const device_info_enum = cl.device.Info;

    var device_name_size: usize = undefined;
    try cl.device.getInfo(device, device_info_enum.name, 0, null, &device_name_size);

    const device_name = try allocator.alloc(u8, device_name_size);
    errdefer allocator.free(device_name);

    try cl.device.getInfo(
        device,
        device_info_enum.name,
        device_name_size,
        device_name.ptr,
        null,
    );

    self.device_name = device_name;

    try cl.device.getInfo(
        device,
        device_info_enum.vendor_id,
        @sizeOf(u32),
        &self.device_vendor_id,
        null,
    );

    var device_type: u64 = undefined;
    try cl.device.getInfo(device, device_info_enum.type, @sizeOf(u64), &device_type, null);
    self.device_type = @enumFromInt(device_type);

    try cl.device.getInfo(
        device,
        device_info_enum.local_mem_type,
        @sizeOf(cl.device.LocalMemType),
        &self.local_mem_type,
        null,
    );

    try cl.device.getInfo(
        device,
        device_info_enum.local_mem_size,
        @sizeOf(u64),
        &self.local_mem_size,
        null,
    );

    const vector_types = .{
        device_info_enum.native_vector_width_char,
        device_info_enum.native_vector_width_char,
        device_info_enum.native_vector_width_short,
        device_info_enum.native_vector_width_short,
        device_info_enum.native_vector_width_int,
        device_info_enum.native_vector_width_int,
        device_info_enum.native_vector_width_long,
        device_info_enum.native_vector_width_long,
        device_info_enum.native_vector_width_float,
        device_info_enum.native_vector_width_double,
    };

    inline for (&self.vector_widths, vector_types) |*vw, vt| {
        var vector_width: u32 = undefined;
        try cl.device.getInfo(device, vt, @sizeOf(u32), &vector_width, null);
        vw.* = @min(vector_width, 16);
    }

    try cl.device.getInfo(
        device,
        device_info_enum.max_compute_units,
        @sizeOf(u32),
        &self.compute_units,
        null,
    );

    try cl.device.getInfo(
        device,
        device_info_enum.max_work_group_size,
        @sizeOf(u64),
        &self.max_work_group_size,
        null,
    );

    try cl.device.getInfo(
        device,
        device_info_enum.global_mem_cacheline_size,
        @sizeOf(u32),
        &self.cache_line_size,
        null,
    );
}

pub fn init(
    self: *CommandQueue,
    ctx: *const Context,
    device: cl.device.DeviceId,
) !void {
    const cmd: cl.command_queue.CommandQueue = try cl.command_queue.create(ctx.ctx, device, 0);
    errdefer cl.command_queue.release(cmd);

    const allocator = ctx.allocator;

    self.allocator = allocator;
    self.ctx = ctx;
    self.cmd = cmd;
    self.device = device;
    self.wekua_id = 0;

    self.headers = .{};

    const programs = try allocator.alloc(
        ?cl.program.Program,
        KernelsSet.TOTAL_NUMBER_OF_HEADERS,
    );
    errdefer allocator.free(programs);
    self.headers.programs = programs;

    @memset(programs, null);

    self.headers.initialized = true;

    for (&self.kernels) |*k| {
        k.* = .{};
    }

    try self.get_device_info(allocator, device);
}

pub fn initMultiples(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    devices: []cl.device.DeviceId,
) ![]CommandQueue {
    var cmd_created: u32 = 0;
    const command_queues: []CommandQueue = try allocator.alloc(CommandQueue, devices.len);
    errdefer {
        for (command_queues[0..cmd_created]) |*cmd| {
            cmd.deinit();
        }
        allocator.free(command_queues);
    }

    for (devices, command_queues, 0..) |device, *wcmd, index| {
        try wcmd.init(ctx, device);
        wcmd.*.wekua_id = index;
        cmd_created += 1;
    }

    return command_queues;
}

pub fn deinit(self: *CommandQueue) void {
    const allocator = self.allocator;

    const kernels = self.kernels;
    for (&kernels) |*set| {
        if (set.kernels) |cl_kernels| {
            for (cl_kernels) |cl_kernel| {
                if (cl_kernel) |clk| {
                    cl.kernel.release(clk);
                }
            }
            allocator.free(cl_kernels);
        }

        if (set.programs) |cl_programs| {
            for (cl_programs) |cl_program| {
                if (cl_program) |clp| {
                    cl.program.release(clp);
                }
            }
            allocator.free(cl_programs);
        }
    }

    for (self.headers.programs.?) |program| {
        if (program) |v| cl.program.release(v);
    }
    allocator.free(self.headers.programs.?);

    allocator.free(self.device_name);

    cl.command_queue.finish(self.cmd) catch |err| {
        std.debug.panic("An error ocurred while executing clFinish: {s}", .{@errorName(err)});
    };
    cl.command_queue.release(self.cmd);
}

pub fn deinitMultiples(allocator: std.mem.Allocator, command_queues: []CommandQueue) void {
    for (command_queues) |*cmd| {
        cmd.deinit();
    }
    allocator.free(command_queues);
}

pub inline fn typeIsSupported(self: *const CommandQueue, comptime T: type) bool {
    return self.vector_widths[Context.getTypeId(T)] > 0;
}

const CommandQueue = @This();

// Unit Tests
const testing = std.testing;

test "CommandQueue.init - basic initialization with real device" {
    const allocator = testing.allocator;

    // Get a real context with devices like context.zig does
    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    // Test that we can access the first command queue
    try testing.expect(context.command_queues.len > 0);

    const cmd_queue = &context.command_queues[0];

    // Verify basic initialization
    try testing.expect(cmd_queue.ctx == context);
    try testing.expect(cmd_queue.wekua_id == 0);

    // Verify device info was populated
    try testing.expect(cmd_queue.device_name.len > 0);
    try testing.expect(cmd_queue.compute_units > 0);
    try testing.expect(cmd_queue.max_work_group_size > 0);
    try testing.expect(cmd_queue.cache_line_size > 0);

    // Verify vector widths are reasonable
    for (cmd_queue.vector_widths) |vw| {
        try testing.expect(vw <= 16); // Should be clamped to max 16
    }
}

test "CommandQueue.initMultiples - multiple command queues creation" {
    const allocator = testing.allocator;

    // Get platforms and devices
    const platforms = try cl.platform.getAll(allocator);
    defer cl.platform.releaseList(allocator, platforms);

    if (platforms.len == 0) return; // Skip if no platforms available

    for (platforms) |plat| {
        var num_devices: u32 = undefined;
        cl.device.getIds(plat.id.?, cl.device.Type.all, null, &num_devices) catch continue;
        if (num_devices == 0) continue;

        const devices = try allocator.alloc(cl.device.DeviceId, num_devices);
        defer {
            for (devices) |dev| {
                cl.device.release(dev);
            }
            allocator.free(devices);
        }

        try cl.device.getIds(plat.id.?, cl.device.Type.all, devices, null);

        // Create a context for testing
        const cl_ctx = try cl.context.create(null, devices, null, null);

        const context = Context.initFromClContext(allocator, cl_ctx) catch |err| {
            cl.context.release(cl_ctx);
            return err;
        };
        defer context.deinit();

        try testing.expect(context.command_queues.len == devices.len);

        // Verify each command queue has correct wekua_id
        for (context.command_queues, 0..) |cmd_queue, index| {
            try testing.expect(cmd_queue.wekua_id == index);
            try testing.expect(cmd_queue.ctx == context);
            try testing.expect(cmd_queue.device_name.len > 0);
        }
    }
}

test "CommandQueue.get_device_info - device information retrieval" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const cmd_queue = &context.command_queues[0];

    // Test device name is not empty
    try testing.expect(cmd_queue.device_name.len > 0);

    // Test device type is valid
    const valid_types = [_]cl.device.Type{ .cpu, .gpu, .accelerator, .custom, .all };
    var type_is_valid = false;
    for (valid_types) |valid_type| {
        if (cmd_queue.device_type == valid_type) {
            type_is_valid = true;
            break;
        }
    }
    try testing.expect(type_is_valid);

    // Test local memory type is valid
    const valid_mem_types = [_]cl.device.LocalMemType{ .local, .global };
    var mem_type_is_valid = false;
    for (valid_mem_types) |valid_mem_type| {
        if (cmd_queue.local_mem_type == valid_mem_type) {
            mem_type_is_valid = true;
            break;
        }
    }
    try testing.expect(mem_type_is_valid);

    // Test that numeric values are reasonable
    try testing.expect(cmd_queue.local_mem_size >= 0);
    try testing.expect(cmd_queue.compute_units > 0);
    try testing.expect(cmd_queue.max_work_group_size > 0);
    try testing.expect(cmd_queue.cache_line_size > 0);
}

test "CommandQueue.typeIsSupported - type support checking" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const cmd_queue = &context.command_queues[0];

    // Test that at least some basic types are typically supported
    // Most devices support these basic types
    const basic_supported = cmd_queue.typeIsSupported(f32) or
        cmd_queue.typeIsSupported(i32) or
        cmd_queue.typeIsSupported(u32);
    try testing.expect(basic_supported);
}

test "CommandQueue.deinit - proper cleanup" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    // The deinit is tested implicitly through context.deinit()
    // We verify that the structure is properly set up for cleanup
    const cmd_queue = &context.command_queues[0];

    try testing.expect(cmd_queue.device_name.len > 0);
    try testing.expect(cmd_queue.allocator.ptr == allocator.ptr);

    // Test that kernels array is properly initialized
    for (cmd_queue.kernels) |kernel_set| {
        // Initially, kernels should be uninitialized
        try testing.expect(kernel_set.kernels == null);
        try testing.expect(kernel_set.programs == null);
        try testing.expect(!kernel_set.initialized);
    }

    // Test that headers are properly initialized
    try testing.expect(cmd_queue.headers.initialized);
    try testing.expect(cmd_queue.headers.programs != null);
    try testing.expect(cmd_queue.headers.programs.?.len == KernelsSet.TOTAL_NUMBER_OF_HEADERS);
}

test "CommandQueue vector widths clamping" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const cmd_queue = &context.command_queues[0];

    // Verify that all vector widths are clamped to maximum 16
    for (cmd_queue.vector_widths) |vw| {
        try testing.expect(vw <= 16);
        try testing.expect(vw >= 0);
    }
}

test "CommandQueue multiple contexts compatibility" {
    const allocator = testing.allocator;

    const contexts = try Context.createOnePerPlatform(allocator, null, cl.device.Type.all);
    defer Context.deinitMultiples(allocator, contexts);

    if (contexts.len == 0) return; // Skip if no contexts available

    // Test that each context has properly initialized command queues
    for (contexts) |context| {
        try testing.expect(context.command_queues.len > 0);

        for (context.command_queues, 0..) |cmd_queue, index| {
            try testing.expect(cmd_queue.wekua_id == index);
            try testing.expect(cmd_queue.ctx == context);
            try testing.expect(cmd_queue.device_name.len > 0);
            try testing.expect(cmd_queue.allocator.ptr == allocator.ptr);
        }
    }
}
