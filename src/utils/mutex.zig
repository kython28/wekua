const std = @import("std");

pub const wMutex = struct {
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    pub fn trylock(self: *wMutex) bool {
        const err: std.c.E = std.c.pthread_mutex_trylock(&self.mutex);
        return switch (err) {
            .BUSY => true,
            else => false
        };
    }

    pub fn lock(self: *wMutex) void {
        const err: std.c.E = std.c.pthread_mutex_lock(&self.mutex);
        if (err != .SUCCESS) unreachable;
    }

    pub fn unlock(self: *wMutex) void {
        const err: std.c.E = std.c.pthread_mutex_unlock(&self.mutex);
        if (err != .SUCCESS) unreachable;
    }

    pub fn destroy(self: *wMutex) void {
        const err = std.c.pthread_mutex_destroy(&self.mutex);
        if (err != .SUCCESS) unreachable;
    }
};
