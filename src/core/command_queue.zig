const std = @import("std");
const cl = @import("opencl");

const Context = @import("context.zig");
const KernelsSet = @import("kernel.zig");

allocator: std.mem.Allocator,
ctx: *const Context,
cmd: cl.command_queue.cl_command_queue,
device: cl.device.cl_device_id,

device_name: []u8,
device_vendor_id: u32,
device_type: cl.device.enums.device_type,

kernels: [KernelsSet.total_number_of_kernels]KernelsSet,
headers: KernelsSet,

local_mem_type: cl.device.enums.local_mem_type,
local_mem_size: u64,

compute_units: u32,
vector_widths: [10]u32,
max_work_group_size: u64,
cache_line_size: u32,
wekua_id: usize,

fn get_device_info(self: *CommandQueue, allocator: std.mem.Allocator, device: cl.device.cl_device_id) !void {
    const device_info_enum = cl.device.enums.device_info;

    var device_name_size: usize = undefined;
    try cl.device.get_info(device, device_info_enum.name, 0, null, &device_name_size);

    const device_name = try allocator.alloc(u8, device_name_size);
    errdefer allocator.free(device_name);

    try cl.device.get_info(device, device_info_enum.name, device_name_size, device_name.ptr, null);

    self.device_name = device_name;

    try cl.device.get_info(device, device_info_enum.vendor_id, @sizeOf(u32), &self.device_vendor_id, null);

    var device_type: u64 = undefined;
    try cl.device.get_info(device, device_info_enum.type, @sizeOf(u64), &device_type, null);
    self.device_type = @enumFromInt(device_type);

    try cl.device.get_info(
        device,
        device_info_enum.local_mem_type,
        @sizeOf(cl.device.enums.local_mem_type),
        &self.local_mem_type,
        null,
    );

    try cl.device.get_info(
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
        try cl.device.get_info(device, vt, @sizeOf(u32), &vector_width, null);
        vw.* = @min(vector_width, 16);
    }

    try cl.device.get_info(device, device_info_enum.max_compute_units, @sizeOf(u32), &self.compute_units, null);

    try cl.device.get_info(
        device,
        device_info_enum.max_work_group_size,
        @sizeOf(u64),
        &self.max_work_group_size,
        null,
    );

    try cl.device.get_info(
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
    device: cl.device.cl_device_id,
) !void {
    const cmd: cl.command_queue.cl_command_queue = try cl.command_queue.create(ctx.ctx, device, 0);
    errdefer cl.command_queue.release(cmd);

    const allocator = ctx.allocator;

    self.allocator = allocator;
    self.ctx = ctx;
    self.cmd = cmd;
    self.device = device;
    self.wekua_id = 0;

    self.headers = .{};

    self.headers.programs = try allocator.alloc(?cl.program.cl_program, KernelsSet.total_number_of_headers);
    errdefer allocator.free(self.headers.programs);

    @memset(self.headers.programs, null);

    self.headers.initialized = true;

    for (&self.kernels) |*k| {
        k.* = .{};
    }

    try self.get_device_info(allocator, device);
}

pub fn init_multiples(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    devices: []cl.device.cl_device_id,
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
        for (set.kernels) |cl_kernel| {
            if (cl_kernel) |clk| {
                cl.kernel.release(clk);
            }
        }
        allocator.free(set.kernels);

        for (set.programs) |cl_program| {
            if (cl_program) |clp| {
                cl.program.release(clp);
            }
        }
        allocator.free(set.programs);
    }

    for (self.headers.programs) |program| {
        if (program) |v| cl.program.release(v);
    }
    allocator.free(self.headers.programs);

    allocator.free(self.device_name);

    cl.command_queue.finish(self.cmd) catch |err| {
        std.debug.panic("An error ocurred while executing clFinish: {s}", .{@errorName(err)});
    };
    cl.command_queue.release(self.cmd);
}

pub fn deinit_multiples(allocator: std.mem.Allocator, command_queues: []CommandQueue) void {
    for (command_queues) |*cmd| {
        cmd.deinit();
    }
    allocator.free(command_queues);
}

pub inline fn typeIsSupported(self: *const CommandQueue, comptime T: type) bool {
    return self.vector_widths[Context.getTypeId(T)] > 0;
}

const CommandQueue = @This();
