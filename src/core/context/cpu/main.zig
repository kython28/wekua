const std = @import("std");
const builtin = @import("builtin");

const Context = @import("../main.zig");
const CommandQueue = @import("../../command_queue.zig");

const Workers = @import("workers.zig");
const CpuCommandQueue = @import("command_queue.zig");
const cpu_buffer = @import("buffer.zig");

pub const Config = struct {
    allocator: std.mem.Allocator,
    buffer_allocator: ?std.mem.Allocator = null,
    workers_count: ?u16 = null,
    work_slots: u16 = 4096,
};

pub const Error = error {
    UnableDetectTarget,
    UnableSetupWorkers,
    InvalidConfig,
} || std.mem.Allocator.Error || std.Io.Cancelable;

allocator: std.mem.Allocator,
buffer_allocator: std.mem.Allocator,
workers: Workers,
target: std.Target,
io: std.Io,

pub fn init(io: std.Io, config: Config) Error!*CpuContext {
    const ctx = try config.allocator.create(CpuContext);
    errdefer config.allocator.destroy(ctx);

    ctx.* = CpuContext{
        .allocator = config.allocator,
        .buffer_allocator = config.buffer_allocator orelse config.allocator,
        .workers = undefined,
        .target = detectTarget(io) catch |e| switch (e) {
            error.Canceled => return error.Canceled,
            else => return e,
        },
        .io = io,
    };

    try ctx.workers.init(config.allocator, config.workers_count, config.work_slots);
    errdefer ctx.workers.deinit();

    return Context{
        .ptr = @ptrCast(ctx),
        .vtable = Context.VTable{
            .alloc = alloc,
            .free = free,
            .createCommandQueue = createCommandQueue,
            .deinit = deinit,
        },
    };
}

fn deinit(ptr: *anyopaque) void {
    const self: *CpuContext = @ptrCast(@alignCast(ptr));
    self.workers.deinit();
    self.allocator.destroy(self);
}

fn detectTarget(io: std.Io) std.zig.system.DetectError!std.Target {
    const query = std.Target.Query{
        .cpu_model = .native,
        .cpu_arch = builtin.cpu.arch,
        .os_tag = builtin.os.tag,
        .abi = builtin.abi,
    };

    const target = try std.zig.system.resolveTargetQuery(io, query);
    return target;
}


fn alloc(ctx_ptr: *anyopaque, len: usize) ?*anyopaque {
    const self: *CpuContext = @alignCast(@ptrCast(ctx_ptr));
    return cpu_buffer.alloc(self.allocator, len);
}

fn free(ctx_ptr: *anyopaque, buf: *anyopaque) void {
    const self: *CpuContext = @alignCast(@ptrCast(ctx_ptr));
    return cpu_buffer.free(self.allocator, @alignCast(@ptrCast(buf)));
}

fn createCommandQueue(ctx_ptr: *anyopaque) std.mem.Allocator.Error!CommandQueue {
    const self: *CpuContext = @alignCast(@ptrCast(ctx_ptr));
    return CpuCommandQueue.create(self);
}

const CpuContext = @This();
