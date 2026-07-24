const std = @import("std");

const CommandQueue = @import("../../command_queue.zig");
const Command = CommandQueue.Command;

const CpuCommandQueue = @This();

ptr: *anyopaque,
vtable: CommandQueue.VTable,

pub fn create() CommandQueue {
    return CommandQueue{
        .ptr = undefined,
        .vtable = .{
            .deinit = deinit,
            .enqueue = enqueue,
            .wait = wait,
        },
    };
}

fn deinit(cq_ptr: *anyopaque) void {
    _ = cq_ptr;
}

fn enqueue(cq_ptr: *anyopaque, command: Command) CommandQueue.Error!void {
    _ = cq_ptr;
    _ = command;
    return error.OutOfMemory;
}

fn wait(cq_ptr: *anyopaque) void {
    _ = cq_ptr;
}
