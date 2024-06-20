const std = @import("std");
const wMutex = @import("mutex.zig").wMutex;

pub const wCondition = struct {
    cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,

    pub fn signal(self: *wCondition) void {
        const err = std.c.pthread_cond_signal(&self.cond);
        if (err != .SUCCESS) unreachable;
    }

    pub fn broadcast(self: *wCondition) void {
        const err = std.c.pthread_cond_broadcast(&self.cond);
        if (err != .SUCCESS) unreachable;
    }

    pub fn wait(self: *wCondition, mutex: *wMutex) void {
        const err = std.c.pthread_cond_wait(&self.cond, &mutex.mutex);
        if (err != .SUCCESS) unreachable;
    }

    pub fn destroy(self: *wCondition) void {
        const err = std.c.pthread_cond_destroy(&self.cond);
        if (err != .SUCCESS) unreachable;
    }
};
