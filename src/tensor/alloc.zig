const std = @import("std");

const cl = @import("opencl");
const w_empty = @import("empty.zig");

const dtypes = @import("utils/dtypes.zig");
const wTensor = dtypes.wTensor;
const wCreateTensorConfig = dtypes.wCreateTensorConfig;

const w_event = @import("utils/event.zig");
const wContext = @import("../core/context.zig").wContext;

pub fn alloc(context: wContext, shape: []const u64, config: wCreateTensorConfig) !wTensor {
    const tensor = try w_empty.empty(context, shape, config);
    errdefer w_empty.release(tensor);

    const prev_events = w_event.acquire_tensor(tensor, .write);
    defer tensor.mutex.unlock();

    const zero: u64 = 0;
    const command_queue = context.command_queues[0];
    const cmd = command_queue.cmd;

    var new_event: cl.event.cl_event = undefined;
    try cl.buffer.fill(
        cmd, tensor.buffer, &zero, @sizeOf(u64),
        0, tensor.size, prev_events,
        &new_event
    );

    try w_event.register_new_event(command_queue, tensor, null, null, new_event, .write);

    return tensor;
}
