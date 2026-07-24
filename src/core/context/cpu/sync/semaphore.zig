const std = @import("std");
const builtin = @import("builtin");
const futex = @import("futex.zig");

/// Semaphore that tracks both pending tasks and sleeping workers.
///
/// `count` encodes active workers plus pending tasks. `workers` is the
/// threshold: values at or below it mean no pending work, so callers block
/// until `up()` publishes new tasks. `waiters` tracks sleeping workers so
/// `up()` can wake exactly `min(pending_tasks, sleeping_workers)`.
const State = packed struct {
    count: u16,
    waiters: u16,
};

state: std.atomic.Value(u32),
workers: u16,

pub fn init(num_of_workers: u16) Semaphore {
    return Semaphore{
        .state = .init(@bitCast(State{ .count = num_of_workers, .waiters = 0 })),
        .workers = num_of_workers,
    };
}

/// Publish `num` new tasks. Wakes up to `num` sleeping workers.
pub fn up(self: *Semaphore, num: u16) void {
    const workers = self.workers;

    var prev_state = self.state.load(.acquire);
    var workers_to_wakeup: u16 = 0;
    while (true) {
        var state: State = @bitCast(prev_state);
        workers_to_wakeup = @min(state.waiters, num);
        state.waiters -= workers_to_wakeup;

        if (state.count < workers) {
            state.count = workers + num;
        } else {
            state.count += num;
        }

        prev_state = self.state.cmpxchgWeak(prev_state, @bitCast(state), .acq_rel, .acquire) orelse break;
    }

    if (workers_to_wakeup > 0) {
        futex.wake(&self.state, @intCast(workers_to_wakeup));
    }
}

/// Wait for a task. Returns immediately when pending work exists, otherwise
/// registers the caller as a waiter and blocks.
pub fn down(self: *Semaphore) void {
    const dec_counter: u32 = @bitCast(State{ .count = 1, .waiters = 0 });

    var prev_state: State = @bitCast(self.state.rmw(.Sub, dec_counter, .acq_rel));

    const workers = self.workers;
    while (prev_state.count <= workers) {
        var new_state = prev_state;
        new_state.waiters += 1;

        // Use cmpxchgStrong because spurious failure here would sleep without
        // registering as a waiter, causing a lost wakeup.
        prev_state = self.state.cmpxchgStrong(
            @bitCast(prev_state),
            @bitCast(new_state),
            .acq_rel,
            .acquire,
        ) orelse blk: {
            futex.wait(&self.state, @bitCast(new_state));
            break :blk @bitCast(self.state.load(.acquire));
        };

        while (prev_state.count > workers) {
            new_state = prev_state;
            new_state.count -= 1;

            prev_state = self.state.cmpxchgWeak(
                @bitCast(prev_state),
                @bitCast(new_state),
                .acq_rel,
                .acquire,
            ) orelse return;
        }
    }
}

const Semaphore = @This();

const testing = std.testing;

test "init initializes count to workers with no waiters" {
    var sem = Semaphore.init(4);
    const state: State = @bitCast(sem.state.load(.acquire));
    try testing.expectEqual(@as(u16, 4), state.count);
    try testing.expectEqual(@as(u16, 0), state.waiters);
}

test "up increments count by num" {
    var sem = Semaphore.init(2);
    sem.up(3);
    const state: State = @bitCast(sem.state.load(.acquire));
    try testing.expectEqual(@as(u16, 5), state.count);
    try testing.expectEqual(@as(u16, 0), state.waiters);
}

test "down returns immediately when work is available" {
    var sem = Semaphore.init(2);
    sem.up(2);
    sem.down();
    const state: State = @bitCast(sem.state.load(.acquire));
    try testing.expectEqual(@as(u16, 3), state.count);
    try testing.expectEqual(@as(u16, 0), state.waiters);
}

test "single up wakes single sleeping worker" {
    if (builtin.single_threaded) {
        return error.SkipZigTest;
    }

    const Ctx = struct {
        sem: *Semaphore,
        started: std.atomic.Value(bool),
        woken: std.atomic.Value(bool),

        fn waiter(self: *@This()) void {
            self.started.store(true, .release);
            self.sem.down();
            self.woken.store(true, .release);
        }
    };

    var sem = Semaphore.init(1);
    var ctx: Ctx = .{
        .sem = &sem,
        .started = std.atomic.Value(bool).init(false),
        .woken = std.atomic.Value(bool).init(false),
    };

    const thread = try std.Thread.spawn(.{}, Ctx.waiter, .{&ctx});

    while (!ctx.started.load(.acquire)) {
        std.Thread.yield() catch {};
    }

    sem.up(1);

    thread.join();
    try testing.expect(ctx.woken.load(.acquire));
}

test "workers process all published tasks" {
    if (builtin.single_threaded) {
        return error.SkipZigTest;
    }

    const num_workers = 3;
    const tasks_per_worker = 10;
    const num_tasks = num_workers * tasks_per_worker;

    const Ctx = struct {
        sem: Semaphore,
        processed: std.atomic.Value(usize),

        fn worker(self: *@This()) void {
            for (0..tasks_per_worker) |_| {
                self.sem.down();
                _ = self.processed.fetchAdd(1, .monotonic);
            }
        }
    };

    var ctx: Ctx = .{
        .sem = Semaphore.init(num_workers),
        .processed = std.atomic.Value(usize).init(0),
    };

    for (0..num_tasks) |_| {
        ctx.sem.up(1);
    }

    var threads: [num_workers]std.Thread = undefined;
    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Ctx.worker, .{&ctx});
    }

    for (threads) |t| t.join();
    try testing.expectEqual(@as(usize, num_tasks), ctx.processed.load(.acquire));
}
