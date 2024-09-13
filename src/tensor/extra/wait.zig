const std = @import("std");
const cl = @import("opencl");

const w_command_queue = @import("../../core/command_queue.zig");
const wCommandQueue = w_command_queue.wCommandQueue;

const w_kernel = @import("../../core/kernel.zig");
const w_event = @import("../utils/event.zig");

const dtypes = @import("../utils/dtypes.zig");
const wTensor = dtypes.wTensor;

const utils = @import("io/utils.zig");

pub fn wait(tensor: wTensor) !void {
    const prev_events = w_event.acquire_tensor(tensor, .read);

    const tensor_mutex = &tensor.mutex;
    defer tensor_mutex.unlock();

    if (prev_events) |events| {
        const te: w_event.wTensorEvent = @alignCast(@ptrCast(tensor.events.last.?.data.?));
        if (te.events_finalized == events.len) return;

        var cond = std.Thread.Condition{};
        try te.callbacks.append(&utils.signal_condition_callback);
        errdefer {
            _ = te.callbacks.pop();
        }
        try te.user_datas.append(&cond);

        cond.wait(tensor_mutex);
    }
}
