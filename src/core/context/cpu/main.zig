const std = @import("std");
const builtin = @import("builtin");

const Context = @import("../main.zig");
const CommandQueue = @import("../../command_queue.zig");

const Workers = @import("workers.zig");
const CpuCommandQueue = @import("command_queue.zig");
const CpuBuffer = @import("buffer.zig");

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
} || std.mem.Allocator.Error;

allocator: std.mem.Allocator,
buffer_allocator: std.mem.Allocator,
workers: Workers,
target: std.Target,

pub fn init(config: Config) Error!*CpuContext {
    const ctx = try config.allocator.create(CpuContext);
    errdefer config.allocator.destroy(ctx);

    ctx.* = CpuContext{
        .allocator = config.allocator,
        .buffer_allocator = config.buffer_allocator orelse config.allocator,
        .workers = undefined,
        .target = detectTarget(config.allocator) catch return error.UnableDetectTarget,
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

fn detectTarget(allocator: std.mem.Allocator) std.zig.system.DetectError!std.Target {
    var pool = std.Io.Threaded.init(allocator);
    defer pool.deinit();

    const io = pool.io();

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
    _ = ctx_ptr;
    _ = len;
    return null;
}

fn free(ctx_ptr: *anyopaque, buf: *anyopaque, len: usize) void {
    _ = ctx_ptr;
    _ = buf;
    _ = len;
}

fn createCommandQueue(ctx_ptr: *anyopaque) CommandQueue {
    _ = ctx_ptr;
    return CpuCommandQueue.create();
}

const CpuContext = @This();
