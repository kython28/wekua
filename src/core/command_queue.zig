const std = @import("std");
const cl = @import("opencl");

const kernel = @import("kernel.zig");

pub const errors = error {
    CommandQueueEventsCounterAlreadyInZero
};

const _w_command_queue = struct {
    allocator: *const std.mem.Allocator,
    ctx: cl.context.cl_context,
    cmd: cl.command_queue.cl_command_queue,

    device_name: []u8,
    device_vendor_id: u32,
    device_type: cl.device.enums.device_type,

    kernels: [kernel.total_number_of_kernels]?kernel.wKernel,

    local_mem_type: cl.device.cl_device_local_mem_type,
    compute_units: u32,
    vector_widths: [10]u32,
    max_work_group_size: u64,

    max_number_of_events: u32,
    current_number_of_events: u32,
    mutex: *std.Thread.Mutex,
    condition: *std.Thread.Condition
};

pub const wCommandQueue = *_w_command_queue;

fn get_device_info(allocator: *const std.mem.Allocator, cmd: wCommandQueue, device: cl.device.cl_device_id) !void {
    const device_info_enum = cl.device.enums.device_info;

    var device_name_size: usize = undefined;
    try cl.device.get_info(
        device, device_info_enum.name, 0, null, &device_name_size
    );

    const device_name = try allocator.alloc(u8, device_name_size);
    errdefer allocator.free(device_name);

    try cl.device.get_info(
        device, device_info_enum.name, device_name_size, device_name.ptr, null
    );

    cmd.device_name = device_name;

    try cl.device.get_info(
        device, device_info_enum.vendor_id, @sizeOf(u32), &cmd.device_vendor_id, null
    );

    var device_type: u64 = undefined;
    try cl.device.get_info(
        device, device_info_enum.vendor_id, @sizeOf(u64), &device_type, null
    );
    cmd.device_type = @enumFromInt(device_type);

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

    try cl.device.get_info(
        device, device_info_enum.max_on_device_events, @sizeOf(u32), &cmd.max_number_of_events, null
    );
}

pub fn create(
    allocator: *const std.mem.Allocator, cl_ctx: cl.context.cl_context, device: cl.device.cl_device_id
) !wCommandQueue {
    const cmd: cl.command_queue.cl_command_queue = try cl.command_queue.create(
        cl_ctx, device, 0
    );
    errdefer cl.command_queue.release(cmd) catch unreachable;

    const new_wcmd = try allocator.create(_w_command_queue);
    errdefer allocator.destroy(new_wcmd);

    new_wcmd.allocator = allocator;
    new_wcmd.ctx = cl_ctx;
    new_wcmd.cmd = cmd;
    new_wcmd.current_number_of_events = 0;

    const mutex = try allocator.create(std.Thread.Mutex);
    mutex.* = std.Thread.Mutex{};
    new_wcmd.mutex = mutex;
    errdefer allocator.destroy(mutex);

    const condition = try allocator.create(std.Thread.Condition);
    condition.* = std.Thread.Condition{};
    new_wcmd.condition = condition;
    errdefer allocator.destroy(condition);

    new_wcmd.current_number_of_events = 0;

    @memset(&new_wcmd.kernels, null);
    try get_device_info(allocator, new_wcmd, device);

    return new_wcmd;
}

pub fn create_multiple(
    allocator: *const std.mem.Allocator, cl_ctx: cl.context.cl_context, devices: []cl.device.cl_device_id
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

pub fn wait_for_command_queue(command_queue: wCommandQueue) void {
    const mutex = command_queue.mutex;
    const cond = command_queue.condition;

    mutex.lock();
    defer mutex.unlock();

    while (command_queue.current_number_of_events > 0){
        cond.wait(mutex);
    }
}

pub fn inc_event_counter(command_queue: wCommandQueue) void {
    const mutex = command_queue.mutex;
    const cond = command_queue.condition;

    mutex.lock();
    defer mutex.unlock();

    const max_number_of_events = command_queue.max_number_of_events;
    while (command_queue.current_number_of_events == max_number_of_events){
        cond.wait(mutex);
    }

    command_queue.current_number_of_events += 1;
}

pub fn dec_event_counter(command_queue: wCommandQueue) !void {
    const mutex = command_queue.mutex;
    const cond = command_queue.condition;

    mutex.lock();
    defer mutex.unlock();

    if (command_queue.current_number_of_events == 0) {
        return errors.CommandQueueEventsCounterAlreadyInZero;
    }
    command_queue.current_number_of_events -= 1;
    cond.signal();
}

pub fn release(command_queue: wCommandQueue) void {
    wait_for_command_queue(command_queue);

    cl.command_queue.release(command_queue.cmd) catch unreachable;
    const allocator = command_queue.allocator;

    allocator.destroy(command_queue.mutex);
    allocator.destroy(command_queue.condition);
    allocator.free(command_queue.device_name);
    allocator.destroy(command_queue);
}

pub fn release_multiple(allocator: *const std.mem.Allocator, command_queues: []wCommandQueue) void {
    for (command_queues) |cmd| {
        release(cmd);
    }
    allocator.free(command_queues);
}
