const std = @import("std");
const cl = @import("opencl");

pub fn release_event(event: cl.event.cl_event, err: anyerror) void {
    cl.event.wait(event) catch |err2| {
        std.debug.panic(
            "An error ocurred ({s}) while waiting for new event and dealing with another error ({s})", .{
                @errorName(err2), @errorName(err)
            }
        );
    };
    cl.event.release(event);
}
