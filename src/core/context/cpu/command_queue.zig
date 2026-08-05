const std = @import("std");

const CommandQueue = @import("../../command_queue.zig");
pub const Command = CommandQueue.Command;
pub const CommnadTag = CommandQueue.CommandTag;
pub const ReadCommand = CommandQueue.ReadCommand;
pub const WriteCommand = CommandQueue.WriteCommand;
pub const ReadRectCommand = CommandQueue.ReadRectCommand;
pub const WriteRectCommand = CommandQueue.WriteRectCommand;
pub const CopyCommand = CommandQueue.CopyCommand;
pub const CopyRectCommand = CommandQueue.CopyRectCommand;
pub const FillCommand = CommandQueue.FillCommand;
pub const FillRectCommand = CommandQueue.FillRectCommand;
pub const EnqueueError = CommandQueue.Error;

const Buffer = @import("buffer.zig");

const Context = @import("main.zig");
const Workers = @import("workers.zig");

const PendingWork = struct {
    slots: []const Workers.Slot,
    offset: u16,
    next: ?*PendingWork = null,
    prev: ?*PendingWork = null,

    pub fn init(
        allocator: std.mem.Allocator,
        offset: u16,
        slots: []const Workers.Slot,
    ) std.mem.Allocator.Error!*PendingWork {
        const work = try allocator.create(PendingWork);
        errdefer allocator.destroy(work);

        work.* = PendingWork{
            .slots = try allocator.dupe(allocator, Workers.Slot, slots),
            .offset = offset,
        };

        return work;
    }

    pub fn deinit(self: *PendingWork, allocator: std.mem.Allocator) void {
        allocator.free(self.slots);
    }

    pub inline fn get(self: *PendingWork) []const Workers.Slot {
        return self.slots[self.offset..];
    }

    pub inline fn increaseOffset(self: *PendingWork, inc: u16) bool {
        const new_offset = self.offset + inc;
        self.offset = new_offset;

        return (@as(usize, @intCast(new_offset)) == self.slots.len);
    }
};

const PendingWorkList = struct {
    head: ?*PendingWork = null,
    tail: ?*PendingWork = null,

    mutex: std.Io.Mutex = .{},
    cond: std.Io.Condition = .{},

    pub inline fn isEmpty(self: *PendingWorkList) bool {
        return (self.head == null);
    }

    pub fn append(self: *PendingWorkList, node: *PendingWork) void {
        const tail = self.tail;
        if (tail) |n| {
            n.next = node;
        }

        node.prev = tail;

        self.tail = node;
        if (tail == self.head) {
            self.head = node;
        }
    }

    pub fn popleft(self: *PendingWorkList, allocator: std.mem.Allocator) void {
        const head = self.head orelse return;

        self.head = head.next;
        if (head == self.tail) {
            self.tail = null;
        }

        head.deinit(allocator);
        allocator.destroy(head);
    }
};

allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
pending_work: PendingWorkList,
slots_to_process: std.atomic.Value(u16) align(std.atomic.cache_line),
workers: *Workers,
io: std.Io,

pub fn create(ctx: *Context) std.mem.Allocator.Error!CommandQueue {
    const allocator = ctx.allocator;

    const cmd = try allocator.create(CpuCommandQueue);
    errdefer allocator.free(cmd);

    cmd.allocator = allocator;
    cmd.arena = .init(allocator);
    cmd.pending_work = PendingWorkList{};
    cmd.workers = &ctx.workers;
    cmd.io = ctx.io;

    ctx.workers.acquire() catch return std.mem.allocator.Error.OutOfMemory;

    return CommandQueue{
        .ptr = @ptrCast(cmd),
        .vtable = .{
            .deinit = deinit,
            .enqueue = enqueue,
            .wait = wait,
        },
    };
}

pub inline fn decreaseCounter(self: *CpuCommandQueue) void {
    const prev_index = self.slots_to_process.rmw(.Sub, 1, .acq_rel);
    if (prev_index == 1) {
        @branchHint(.unlikely);
        self.pushPendingWorker();
    }
}

fn pushPendingWorker(self: *CpuCommandQueue) void {
    self.pending_work.mutex.lockUncancelable(self.io);
    defer self.pending_work.mutex.unlock(self.io);

    const head = self.pending_work.head orelse {
        self.pending_work.cond.broadcast(self.io);
        return;
    };
    const slots = head.get();

    const workers = self.workers;
    const pushed_slots = workers.push(slots);
    self.slots_to_process.store(pushed_slots, .release);
    workers.wakeup(pushed_slots);

    if (head.increaseOffset(pushed_slots)) {
        @branchHint(.unlikely);
        self.pending_work.popleft(self.allocator);
    }
}

fn deinit(cq_ptr: *anyopaque) void {
    const self: *CpuCommandQueue = @ptrCast(@alignCast(cq_ptr));
    self.waitUncancelable();
    self.arena.deinit();

    const allocator = self.allocator;
    allocator.destroy(self);
}

fn getSlots(
    self: *CpuCommandQueue,
    allocator: std.mem.Allocator,
    command: Command,
) CommandQueue.Error![]const Workers.Slot {
    return switch (command) {
        .read => |c| try Buffer.prepareReadCommand(self, allocator, c),
        .write => |c| try Buffer.prepareWriteCommand(self, allocator, c),
        .read_rect => |c| try Buffer.prepareReadRectCommand(self, allocator, c),
        .write_rect => |c| try Buffer.prepareWriteRectCommand(self, allocator, c),
        .copy => |c| try Buffer.prepareCopyCommand(self, allocator, c),
        .copy_rect => |c| try Buffer.prepareCopyRectCommand(self, allocator, c),
        .fill => |c| try Buffer.prepareFillCommand(self, allocator, c),
        .fill_rect => |c| try Buffer.prepareFillRectCommand(self, allocator, c),
    };
}

fn enqueue(cq_ptr: *anyopaque, command: Command) CommandQueue.Error!void {
    const self: *CpuCommandQueue = @ptrCast(@alignCast(cq_ptr));
    const arena_allocator = self.arena.allocator();
    defer _ = self.arena.reset(.retain_capacity);

    const allocator = self.allocator;

    const slots = try self.getSlots(command, arena_allocator);

    try self.pending_work.mutex.lock(self.io);
    defer self.pending_work.mutex.unlock();

    var pushed_slots: u16 = 0;
    if (self.pending_work.isEmpty()) {
        const workers = self.workers;
        pushed_slots = workers.push(slots);
        if (pushed_slots == 0) {
            @panic("Work dispatcher is full. Command Queue's reserved slot is busy");
        }

        self.slots_to_process.store(pushed_slots, .release);
        workers.wakeup(pushed_slots);

        if (pushed_slots == slots.len) {
            return;
        }
    }

    const work = try PendingWork.init(allocator, pushed_slots, slots);
    self.pending_work.append(work);
}

fn wait(cq_ptr: *anyopaque) std.Io.Cancelable!void {
    const self: *CpuCommandQueue = @ptrCast(@alignCast(cq_ptr));

    try self.pending_work.mutex.lock(self.io);
    defer self.pending_work.mutex.unlock(self.io);

    while (!self.pending_work.isEmpty()) {
        try self.pending_work.cond.wait(self.io, &self.pending_work.mutex);
    }
}

fn waitUncancelable(self: *CpuCommandQueue) void {
    self.pending_work.mutex.lockUncancelable();
    defer self.pending_work.mutex.unlock();

    while (!self.pending_work.isEmpty()) {
        self.pending_work.cond.waitUncancelable(self.io, &self.pending_work.mutex);
    }
}

const CpuCommandQueue = @This();
