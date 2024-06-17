const w_empty = @import("empty.zig");
const wTensor = w_empty.wTensor;
const wCreateTensorConfig = w_empty.wCreateTensorConfig;

const w_event = @import("event.zig");
const cl = @import("opencl");
const wContext = @import("../core/context.zig").wContext;

pub fn alloc(context: wContext, shape: []const u64, config: wCreateTensorConfig) !wTensor {
    const tensor = try w_empty.empty(context, shape, config);

    const prev_event = w_event.acquire_tensor(tensor);
    defer tensor.mutex.unlock();

    const zero: u64 = 0;
    const command_queue = context.command_queues[0];
    const cmd = command_queue.cmd;

    var new_event: cl.event.cl_event = undefined;
    var events_to_wait: ?[]const cl.event.cl_event = null;
    if (prev_event) |e| {
        events_to_wait = @as([*]const cl.event.cl_event, @ptrCast(&e))[0..1];
    }
    try cl.buffer.fill(
        cmd, tensor.buffer, &zero, @sizeOf(u64),
        0, tensor.size, events_to_wait,
        &new_event
    );

    try w_event.register_new_event(command_queue, tensor, new_event, .write, true);

    return tensor;
}
