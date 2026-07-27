const std = @import("std");

const CommandQueue = @import("../../command_queue.zig");
pub const Command = CommandQueue.Command;
pub const ReadCommand = CommandQueue.ReadCommand;
pub const WriteCommand = CommandQueue.WriteCommand;
pub const ReadRectCommand = CommandQueue.ReadRectCommand;
pub const WriteRectCommand = CommandQueue.WriteRectCommand;
pub const CopyCommand = CommandQueue.CopyCommand;
pub const CopyRectCommand = CommandQueue.CopyRectCommand;
pub const FillCommand = CommandQueue.FillCommand;
pub const FillRectCommand = CommandQueue.FillRectCommand;
pub const EnqueueError = CommandQueue.Error;

const Context = @import("main.zig");
const Workers = @import("workers.zig");

const sync = @import("sync/main.zig");
const Mutex = sync.Mutex;


const PendingWork = struct {
    slots: []const Workers.Slot,
    offset: u16,
    next: ?*PendingWork,
    prev: ?*PendingWork,

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

    mutex: Mutex = .{},

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


pub fn create(ctx: *Context) CommandQueue {
    const allocator = ctx.allocator;

    const cmd = try allocator.create(CpuCommandQueue);
    errdefer allocator.free(cmd);

    cmd.allocator = allocator;
    cmd.arena = .init(allocator);
    cmd.pending_work = PendingWorkList{};
    cmd.workers = &ctx.workers;

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
    self.pending_work.mutex.lock();
    defer self.pending_work.mutex.unlock();

    const head = self.pending_work.head orelse return;
    const slots = head.get();

    const workers = self.workers;
    const pushed_slots = workers.push(true, slots);
    self.slots_to_process.store(pushed_slots, .release);
    workers.wakeup(pushed_slots);

    if (head.increaseOffset(pushed_slots)) {
        @branchHint(.unlikely);
        self.pending_work.popleft(self.allocator);
    }
}

fn deinit(cq_ptr: *anyopaque) void {
    _ = cq_ptr;
}

fn enqueue(cq_ptr: *anyopaque, command: Command) CommandQueue.Error!void {
    const self: *CpuCommandQueue = @alignCast(@ptrCast(cq_ptr));

    // const slots = switch (command) {
    //     .read => |data| 
    // };
}

fn wait(cq_ptr: *anyopaque) void {
    _ = cq_ptr;
}

const CpuCommandQueue = @This();
