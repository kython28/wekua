const std = @import("std");
const builtin = @import("builtin");

const sync = @import("sync/main.zig");
const Semaphore = sync.Semaphore;
const Mutex = sync.Mutex;

const CommandQueue = @import("command_queue.zig");

const c = @cImport({
    switch (builtin.os.tag) {
        .linux => @cInclude("sched.h"),
        .freebsd, .dragonfly, .netbsd => @cInclude("cpuset.h"),
        .windows => @cInclude("windows.h"),
        else => {},
    }
});

/// Error set for operations that bind a worker thread to a specific CPU.
const AffinityError = std.posix.UnexpectedError || std.posix.SchedError ||
    std.posix.PthreadError || std.os.windows.GetLastErrorError ||
    error{
        AffinityNotSupported,
        InvalidCpuIndex,
        OutOfMemory,
    };

/// Bind `thread` to the CPU identified by `cpu`.
///
/// The implementation uses the platform-specific thread-affinity API:
/// * Linux: `sched_setaffinity` against the kernel thread id returned by the
///   thread handle.
/// * FreeBSD / DragonFly: `cpuset_setaffinity` for the current thread's TID.
/// * NetBSD: dynamically-sized `cpuset_t` allocated on demand and passed to
///   `sched_setaffinity`.
/// * Windows: `SetThreadAffinityMask` using a single-bit mask.
///
/// On unsupported operating systems the call returns without doing anything.
pub fn setThreadAffinity(thread: *std.Thread, cpu: usize) AffinityError!void {
    switch (builtin.os.tag) {
        .linux => {
            // Use sched_setaffinity() against the kernel thread id (gettid).
            // std.Thread.getHandle() on Linux returns that tid directly.
            var set: c.cpu_set_t = std.mem.zeroes(c.cpu_set_t);
            c.CPU_SET(@intCast(cpu), &set);

            const tid: c.pid_t = @intCast(thread.getHandle());
            switch (std.posix.errno(c.sched_setaffinity(
                tid,
                @sizeOf(c.cpu_set_t),
                &set,
            ))) {
                .SUCCESS => return,
                .INVAL => return error.InvalidCpuIndex,
                .PERM => return error.PermissionDenied,
                .SRCH => unreachable,
                else => |e| return std.posix.unexpectedErrno(e),
            }
        },

        .freebsd, .dragonfly => {
            // cpuset_setaffinity() with CPU_LEVEL_WHICH / CPU_WHICH_TID
            // targets the calling thread when id == -1.
            var set: c.cpuset_t = std.mem.zeroes(c.cpuset_t);
            c.CPU_SET(@intCast(cpu), &set);

            switch (std.posix.errno(c.cpuset_setaffinity(
                c.CPU_LEVEL_WHICH,
                c.CPU_WHICH_TID,
                -1,
                @sizeOf(c.cpuset_t),
                &set,
            ))) {
                .SUCCESS => return,
                .INVAL => return error.InvalidCpuIndex,
                .DEADLK => return error.InvalidCpuIndex,
                .PERM => return error.PermissionDenied,
                .SRCH => unreachable,
                else => |e| return std.posix.unexpectedErrno(e),
            }
        },

        .netbsd => {
            // NetBSD's cpuset_t is an opaque, dynamically-sized type.
            // We first ask the kernel how big the mask must be, then allocate
            // it on the heap, set the bit and hand it to sched_setaffinity().
            var size: c.size_t = 0;
            _ = c.cpuset_getaffinity(
                c.CPU_LEVEL_WHICH,
                c.CPU_WHICH_TID,
                -1,
                0,
                null,
                &size,
            );
            if (size == 0) return error.AffinityNotSupported;

            const buf = try std.heap.page_allocator.alignedAlloc(u8, .@"4", size);
            defer std.heap.page_allocator.free(buf);
            @memset(buf, 0);

            c.cpuset_set(@intCast(cpu), buf.ptr);

            switch (std.posix.errno(c.sched_setaffinity(
                @intCast(@intFromPtr(thread.getHandle())),
                size,
                @ptrCast(buf.ptr),
            ))) {
                .SUCCESS => return,
                .INVAL => return error.InvalidCpuIndex,
                .PERM => return error.PermissionDenied,
                .SRCH => unreachable,
                else => |e| return std.posix.unexpectedErrno(e),
            }
        },

        .windows => {
            const mask: std.os.windows.DWORD_PTR =
                @as(std.os.windows.DWORD_PTR, 1) << @intCast(cpu);
            const prev = c.SetThreadAffinityMask(thread.getHandle(), mask);
            if (prev == 0) {
                const err = std.os.windows.GetLastError();
                switch (err) {
                    .INVALID_HANDLE => unreachable,
                    .INVALID_PARAMETER => return error.InvalidCpuIndex,
                    else => return std.os.windows.unexpectedError(err),
                }
            }
        },
        else => {},
    }
}

/// Maximum number of `usize` arguments stored inside a single work `Slot`.
const MAX_ARGS = 30;

/// Unit of work queued to the worker pool.
///
/// A slot contains the function to execute and the raw `usize` arguments that
/// will be passed to it. The caller is responsible for ensuring the argument
/// layout matches what `callback` expects.
pub const Slot = struct {
    /// Function called by the worker thread. A null callback signals the worker
    /// to shut down.
    callback: ?*const fn ([]const usize) void,

    command_queue: *CommandQueue,

    /// Argument buffer passed verbatim to `callback`.
    args: [MAX_ARGS]usize,
};

workers: []std.Thread,
slots: []Slot,

/// Lock-free semaphore used to park workers and wake them when new work is
/// published. It is backed by a single atomic `u32` and a futex. The initial
/// worker count is also the number of slots reserved for the workers
/// themselves (see `reserverd_slots`).
sem: Semaphore,

/// Mutex that protects the producer side of the ring buffer: `producer_index`,
/// `reserverd_slots`, and the batch push logic.
mutex: Mutex,

/// Index read atomically by workers to claim the next slot. Kept on its own
/// cache line to avoid false sharing.
consumer_index: std.atomic.Value(usize) align(std.atomic.cache_line),

/// Index where the next slot will be written by the producer. Only mutated
/// while holding `mutex`.
producer_index: usize,

/// Number of slots reserved by producers plus the worker threads themselves.
/// Initialized to the worker count so every worker can always be signaled
/// with a shutdown slot. Only mutated while holding `mutex`.
reserverd_slots: u16,


/// Initialize the worker pool.
///
/// `work_slots` must be a power of two; it defines the capacity of the
/// internal ring buffer. `workers_count` selects how many OS threads are
/// spawned; when null the number of online CPUs is used. Each worker is
/// bound to a CPU in round-robin order via `setThreadAffinity`.
///
    /// Requires `work_slots >= workers_count * 2`; this guarantees that every
    /// worker can be issued a reserved shutdown slot without filling the buffer.
    ///
    /// The caller must call `deinit` to join the threads and free the buffers.
pub fn init(
    self: *Workers,
    allocator: std.mem.Allocator,
    workers_count: ?u16,
    work_slots: u16,
) error{ UnableSetupWorkers, OutOfMemory, InvalidConfig }!void {
    if (!std.math.isPowerOfTwo(work_slots)) {
        return error.InvalidConfig;
    }

    const cpu_count = std.Thread.getCpuCount() catch return error.UnableSetupWorkers;
    const count: u16 = workers_count orelse @intCast(cpu_count);

    if (work_slots < count * 2) {
        return error.InvalidConfig;
    }

    const slots = try allocator.alloc(Slot, work_slots);
    errdefer allocator.free(slots);
    self.slots = slots;

    @memset(slots, std.mem.zeroes(Slot));


    self.sem = .init(count);
    self.mutex = Mutex{};
    self.consumer_index = .init(0);
    self.producer_index = 0;
    self.reserverd_slots = count;

    const threads = try allocator.alloc(std.Thread, @intCast(count));
    errdefer allocator.free(threads);

    var threads_started: usize = 0;
    errdefer {
        self.sem.up(@intCast(threads_started));
        for (threads[0..threads_started]) |*t| {
            t.join();
        }
    }

    for (threads, 0..) |*t, i| {
        t.* = std.Thread.spawn(.{}, worker, .{self}) catch return error.UnableSetupWorkers;
        threads_started += 1;

        setThreadAffinity(t, i % cpu_count) catch return error.UnableSetupWorkers;
    }

    self.workers = threads;
}

/// Shut down the worker pool and free its resources.
///
/// Must be called exactly once. It acquires `mutex`, pushes one reserved
/// shutdown slot (a zeroed `Slot`) per worker, wakes them with the lock-free
/// semaphore, then joins every thread and frees the thread and slot buffers
/// using `allocator`.
pub fn deinit(self: *Workers, allocator: std.mem.Allocator) void {
    {
        self.mutex.lock();
        defer self.mutex.unlock();

        const empty_slot = std.mem.zeroes(Slot);
        for (0..self.workers.len) |_| {
            const pushed = self.push(true, &.{empty_slot});
            if (pushed != 1) {
                @panic("Unexpected behavior while shutting down thread pool... Unable to signal shutdown");
            }
        }

        self.sem.up(self.workers.len);
    }

    for (self.workers) |*t| {
        t.join();
    }

    allocator.free(self.workers);
    allocator.free(self.slots);
}

/// Background worker entry point.
///
/// Loops forever waiting on the semaphore, claims the next slot using an
/// atomic fetch-and-add on `consumer_index`, and executes its callback. A
/// slot whose `callback` is null terminates the loop and lets the thread
/// exit.
fn worker(self: *Workers) void {
    const sem = &self.sem;
    const consumer_index = &self.consumer_index;
    const slots = self.slots;

    const mask = slots.len - 1;

    while (true) {
        sem.down();

        const index = consumer_index.rmw(.Add, 1, .acq_rel);
        const slot = &slots[index & mask];

        const callback = slot.callback orelse return;
        callback(&slot.args);

        slot.command_queue.decreaseCounter();
    }
}

/// Acquire the producer-side mutex.
pub fn lock(self: *Workers) void {
    self.mutex.lock();
}

/// Release the producer-side mutex.
pub fn unlock(self: *Workers) void {
    self.mutex.unlock();
}

/// Wake up to `num` sleeping workers using the lock-free semaphore.
///
/// This operation is lock-free and does not require holding `lock`. It is
/// usually called after pushing one or more slots so workers can start
/// consuming them.
pub fn wakeup(self: *Workers, num: u16) void {
    self.sem.up(num);
}

/// Return how many slots are still available in the ring buffer.
///
/// The count is computed from the distance between the atomic consumer
/// index and the local producer index, both masked to the buffer size.
/// When both indices are equal the buffer is considered empty, therefore
/// the full capacity is reported.
pub fn getRemainingCapacity(self: *Workers) u16 {
    const slots_capacity = self.slots.len;
    const mask = slots_capacity - 1;
    var consumer_index = self.consumer_index.load(.acquire) & mask;
    const producer_index = self.producer_index & mask;


    if (consumer_index == producer_index) {
        return slots_capacity;
    }

    if (consumer_index < producer_index) {
        consumer_index += slots_capacity;
    }

    return consumer_index - producer_index + 1;
}

/// Reserve one slot for later use.
///
/// Must be called while holding `lock`. Decreases the available capacity
/// by one so a subsequent `push` can still succeed even if the buffer
/// becomes full.
pub fn acquire(self: *Workers) error{OutOfSlots}!void {
    if (self.getRemainingCapacity() < self.reserverd_slots) {
        return error.OutOfSlots;
    }

    self.reserverd_slots += 1;
}

/// Release a previously reserved slot.
///
/// Must be called while holding `lock`.
pub fn release(self: *Workers) void {
    self.reserverd_slots -= 1;
}

/// Push up to `slots.len` work units into the ring buffer.
///
/// Must be called while holding `lock`. The function copies the slots into
/// the buffer, advances `producer_index`, and returns the number of slots
/// actually pushed.
///
/// * If there is enough free capacity, all `slots` are copied into the buffer.
/// * If capacity is smaller than `slots.len`, only the prefix that fits is copied.
/// * When `use_reserved` is true and at least one reserved slot exists, a
///   single slot is always accepted even if the buffer appears full.
///
/// Returns the number of slots actually pushed. The caller is responsible
/// for waking the corresponding number of workers with `wakeup`.
pub fn push(self: *Workers, use_reserved: bool, slots: []const Slot) u16 {
    const rem_capacity = self.getRemainingCapacity();

    const workers_slots = self.slots;
    const mask = workers_slots.len - 1;
    const producer_index = self.producer_index & mask;

    if (rem_capacity <= self.reserverd_slots) {
        @branchHint(.unlikely);
        if (!use_reserved or rem_capacity == 0) {
            return 0;
        }

        self.slots[producer_index] = slots[0];
        self.producer_index += 1;
        return 1;
    }

    var user_slots = slots;
    if (rem_capacity < slots.len) {
        user_slots = slots[0..rem_capacity];
    }

    if ((producer_index + user_slots.len) > workers_slots.len) {
        const diff = workers_slots.len - producer_index;
        @memcpy(workers_slots[producer_index..], user_slots[0..diff]);
        @memcpy(workers_slots[0..user_slots.len - diff], user_slots[diff..]);
    }else{
        @memcpy(workers_slots[producer_index..producer_index + user_slots.len], user_slots);
    }

    self.producer_index += user_slots.len;
    return @intCast(user_slots.len);
}

const Workers = @This();
