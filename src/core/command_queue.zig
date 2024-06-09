const std = @import("std");
const cl = @import("opencl");

pub const wKernels = enum (u16) {
    Random = 0,
    RandRange = 1,
};

const total_number_of_kernels: u16 = @intCast(@typeInfo(wKernels).Enum.fields.len * 3);

const _w_command_queue = struct {
    allocator: std.mem.Allocator,
    cmd: cl.command_queue.cl_command_queue,

    programs: [total_number_of_kernels]?cl.program.cl_program = .{null} ** total_number_of_kernels,
    kernels: [total_number_of_kernels]?cl.kernel.cl_kernel = .{null} ** total_number_of_kernels,

    local_mem_type: cl.device.cl_device_local_mem_type,
    compute_units: u32,
    vector_widths: [10]u32,
    max_work_group_size: u64
};

pub const wCommandQueue = *_w_command_queue;

fn get_device_info(cmd: wCommandQueue, device: cl.device.cl_device_id) !void {
    const device_info_enum = cl.device.enums.device_info;
    try cl.device.get_info(
        device, device_info_enum.local_mem_type, @sizeOf(cl.device.cl_device_local_mem_type),
        &cmd.local_mem_type, null
    );

    const vector_types = .{
        device_info_enum.preferred_vector_width_char,
        device_info_enum.preferred_vector_width_char,
        device_info_enum.preferred_vector_width_short,
        device_info_enum.preferred_vector_width_short,
        device_info_enum.preferred_vector_width_int,
        device_info_enum.preferred_vector_width_int,
        device_info_enum.preferred_vector_width_long,
        device_info_enum.preferred_vector_width_long,
        device_info_enum.preferred_vector_width_float,
        device_info_enum.preferred_vector_width_double
    };

    inline for (&cmd.vector_widths, vector_types) |*vw, vt| {
        try cl.device.get_info(
            device, vt, @sizeOf(u32), vw, null
        );
    }

    try cl.device.get_info(
        device, device_info_enum.max_compute_units, @sizeOf(u32), &cmd.compute_units, null
    );

    try cl.device.get_info(
        device, device_info_enum.max_work_group_size, @sizeOf(u64), &cmd.max_work_group_size, null
    );
}

pub fn create(
    allocator: std.mem.Allocator, cl_ctx: cl.context.cl_context, device: cl.device.cl_device_id
) !wCommandQueue {
    const cmd: cl.command_queue.cl_command_queue = try cl.command_queue.create(
        cl_ctx, device, 0
    );
    errdefer cl.command_queue.release(cmd) catch unreachable;

    const new_wcmd = try allocator.create(_w_command_queue);
    errdefer allocator.destroy(new_wcmd);

    new_wcmd.cmd = cmd;
    try get_device_info(new_wcmd, device);
    new_wcmd.allocator = allocator;

    return new_wcmd;
}

pub fn create_multiple(
    allocator: std.mem.Allocator, cl_ctx: cl.context.cl_context, devices: []cl.device.cl_device_id
) ![]wCommandQueue {
    var cmd_created: u32 = 0;
    const command_queues: []wCommandQueue = try allocator.alloc(wCommandQueue, devices.len);
    errdefer {
        if (cmd_created > 0) {
            for (command_queues[0..cmd_created]) |cmd| {
                release(cmd);
            }
        }
        allocator.free(command_queues);
    }

    for (devices, command_queues) |device, *wcmd| {
        wcmd.* = try create(allocator, cl_ctx, device);
        cmd_created += 1;
    }

    return command_queues;
}

pub fn release(command_queue: wCommandQueue) void {
    cl.command_queue.release(command_queue.cmd) catch unreachable;
    command_queue.allocator.destroy(command_queue);
}

pub fn release_multiple(allocator: std.mem.Allocator, command_queues: []wCommandQueue) void {
    for (command_queues) |cmd| {
        release(cmd);
    }
    allocator.free(command_queues);
}
