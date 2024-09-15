const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_kernel = @import("../../core/kernel.zig");
const w_event = @import("../utils/event.zig");

const dtypes = @import("../utils/dtypes.zig");
const wTensor = dtypes.wTensor;

const utils = @import("io/utils.zig");

inline fn check(tensor_event: w_event.wTensorEvent) bool {
    return switch (tensor_event.event_type) {
        .read => (tensor_event.events_finalized < tensor_event.read_events.?.items.len),
        .write => (tensor_event.events_finalized < 1)
    };
}

pub fn wait(tensor: wTensor) !void {
    const tensor_mutex = &tensor.mutex;
    const tensor_cond = &tensor.condition;
    tensor_mutex.lock();
    defer tensor_mutex.unlock();

    const te: w_event.wTensorEvent = @alignCast(@ptrCast(tensor.events.last.?.data.?));
    while (check(te)) {
        tensor_cond.wait(tensor_mutex);
    }
}
