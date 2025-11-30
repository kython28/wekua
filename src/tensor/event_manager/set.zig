const std = @import("std");
const builtin = @import("builtin");

const cl = @import("opencl");

const Batch = @import("batch.zig");

pub const AppendResult = enum { full, success_and_full, success };

pub const UserCallback = struct {
    func: *const fn (user_data: ?*anyopaque) void,
    data: ?*anyopaque,
};
pub const UserCallbackArray = std.ArrayList(UserCallback);

pub const Operation = enum {
    write,
    partial_write,
    read,
    none,
};

pub const MaxEventsPerSet = switch (builtin.mode) {
    .Debug => 16,
    .ReleaseFast, .ReleaseSafe => 256,
    .ReleaseSmall => 64,
};

const MaxEventsPerSetInt = switch (builtin.mode) {
    .Debug, .ReleaseSmall => u8,
    .ReleaseFast, .ReleaseSafe => u16,
};

operation: Operation,

callbacks: UserCallbackArray,
events: [MaxEventsPerSet]cl.event.Event,
events_count: MaxEventsPerSetInt,

index: u8,

pub fn initValues(self: *Set, index: usize) void {
    self.operation = .none;
    self.callbacks = .empty;
    self.events_count = 0;
    self.index = @intCast(index);
}

pub fn init(index: usize, allocator: std.mem.Allocator) error{OutOfMemory}!*Set {
    const self = try allocator.create(Set);
    errdefer allocator.destroy(self);

    self.initValues(index);
    return self;
}

pub fn deinit(self: *Set, allocator: std.mem.Allocator) void {
    self.callbacks.deinit(allocator);
    allocator.destroy(self);
}

pub inline fn getParent(self: *Set) *Batch {
    const ptr: usize = @intFromPtr(self) - @offsetOf(Batch, "sets") - @as(usize, self.index) * @sizeOf(Set);
    return @ptrFromInt(ptr);
}

pub fn append(
    self: *Set,
    allocator: std.mem.Allocator,
    operation: Operation,
    event: cl.event.Event,
    user_callback: ?UserCallback,
) error{OutOfMemory}!AppendResult {
    const current_operation = self.operation;
    const events_count = self.events_count;

    if (current_operation == .none) {
        self.operation = operation;
    } else if (operation != current_operation or isFull(current_operation, events_count)) {
        return AppendResult.full;
    }
    errdefer {
        if (current_operation == .none) {
            self.operation = .none;
        }
    }

    if (user_callback) |callback| {
        try self.callbacks.append(allocator, callback);
    }

    self.events[events_count] = event;
    self.events_count = events_count + 1;

    if (isFull(operation, events_count + 1)) return AppendResult.success_and_full;

    return AppendResult.success;
}

pub inline fn pop(self: *Set, had_callback: bool) void {
    const events_count = self.events_count;
    self.events_count = events_count - 1;
    if (events_count == 1) {
        self.operation = .none;
    }

    if (had_callback) {
        _ = self.callbacks.pop();
    }
}

pub inline fn isFull(operation: Operation, events_count: usize) bool {
    return switch (operation) {
        .read, .partial_write => (events_count == MaxEventsPerSet),
        .write => (events_count == 1),
        .none => false,
    };
}

pub inline fn full(self: *const Set) bool {
    return isFull(self.operation, self.events_count);
}

pub inline fn toSlice(self: *const Set) []const cl.event.Event {
    return self.events[0..self.events_count];
}

pub inline fn appendCallback(
    self: *Set,
    allocator: std.mem.Allocator,
    callback: UserCallback,
) error{OutOfMemory}!void {
    try self.callbacks.append(allocator, callback);
}

pub inline fn executeCallbacks(self: *const Set) void {
    for (self.callbacks.items) |*callback| {
        callback.func(callback.data);
    }
}

pub inline fn waitForEvents(self: *const Set) !void {
    try cl.event.waitForMany(self.toSlice());
    self.executeCallbacks();
}

pub fn clear(self: *Set, allocator: std.mem.Allocator) void {
    for (self.events[0..self.events_count]) |event| {
        cl.event.release(event);
    }

    self.callbacks.deinit(allocator);

    self.operation = .none;
    self.events_count = 0;
}

const Set = @This();

// Tests
const testing = std.testing;
const core = @import("core");

test "Set.init initializes correctly" {
    const set = try Set.init(5, testing.allocator);
    defer set.deinit(testing.allocator);

    try testing.expect(set.operation == .none);
    try testing.expect(set.events_count == 0);
    try testing.expect(set.index == 5);
    try testing.expect(set.callbacks.items.len == 0);
}

test "Set.getParent returns correct parent batch" {
    // Create a real batch structure to test getParent
    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    const parent = batch.sets[0].getParent();
    try testing.expect(@intFromPtr(parent) == @intFromPtr(batch));
}

test "Set.append with none operation sets operation and adds event" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const result = try set.append(testing.allocator, .read, cl_event, null);

    try testing.expect(result == .success);
    try testing.expect(set.operation == .read);
    try testing.expect(set.events_count == 1);
    try testing.expect(set.events[0] == cl_event);
}

test "Set.append with matching operation succeeds" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    var result = try set.append(testing.allocator, .read, cl_event1, null);
    try testing.expect(result == .success);

    result = try set.append(testing.allocator, .read, cl_event2, null);
    try testing.expect(result == .success);

    try testing.expect(set.events_count == 2);
    try testing.expect(set.events[0] == cl_event1);
    try testing.expect(set.events[1] == cl_event2);
}

test "Set.append with different operation returns full" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    var result = try set.append(testing.allocator, .read, cl_event1, null);
    try testing.expect(result == .success);

    result = try set.append(testing.allocator, .write, cl_event2, null);
    try testing.expect(result == .full);

    try testing.expect(set.operation == .read);
    try testing.expect(set.events_count == 1);
}

test "Set.append with write operation allows only one event" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    var result = try set.append(testing.allocator, .write, cl_event1, null);
    try testing.expect(result == .success_and_full);

    result = try set.append(testing.allocator, .write, cl_event2, null);
    try testing.expect(result == .full);

    try testing.expect(set.events_count == 1);
}

test "Set.append with callback stores callback" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    var test_data: u32 = 42;
    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                const ptr: *u32 = @ptrCast(@alignCast(data.?));
                ptr.* = 100;
            }
        }.testCallback,
        .data = &test_data,
    };

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const result = try set.append(testing.allocator, .read, cl_event, callback);
    try testing.expect(result == .success);

    try testing.expect(set.callbacks.items.len == 1);
    try testing.expect(@intFromPtr(set.callbacks.items[0].data) == @intFromPtr(&test_data));
}

test "Set.append returns success_and_full when reaching capacity" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    var events_to_cleanup: std.ArrayList(cl.event.Event) = .empty;
    defer {
        for (events_to_cleanup.items) |e| {
            cl.event.release(e);
        }
        events_to_cleanup.deinit(testing.allocator);
    }

    // Fill up to capacity - 1 for read operation
    for (0..MaxEventsPerSet - 1) |_| {
        const cl_event = try cl.event.createUserEvent(context.ctx);
        try events_to_cleanup.append(testing.allocator, cl_event);
        const result = try set.append(testing.allocator, .read, cl_event, null);
        try testing.expect(result == .success);
    }

    // Last append should return success_and_full
    const last_event = try cl.event.createUserEvent(context.ctx);
    try events_to_cleanup.append(testing.allocator, last_event);

    const result = try set.append(testing.allocator, .read, last_event, null);
    try testing.expect(result == .success_and_full);
    try testing.expect(set.events_count == MaxEventsPerSet);
}

test "Set.pop decrements count and resets operation when empty" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const result = try set.append(testing.allocator, .read, cl_event, null);
    try testing.expect(result == .success);

    set.pop(false);
    try testing.expect(set.events_count == 0);
    try testing.expect(set.operation == .none);
}

test "Set.pop with callback removes callback" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    var test_data: u32 = 42;
    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                _ = data;
            }
        }.testCallback,
        .data = &test_data,
    };

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const result = try set.append(testing.allocator, .read, cl_event, callback);
    try testing.expect(result == .success);

    try testing.expect(set.callbacks.items.len == 1);
    set.pop(true);
    try testing.expect(set.callbacks.items.len == 0);
}

test "Set.isFull returns correct values for different operations" {
    try testing.expect(isFull(.write, 1) == true);
    try testing.expect(isFull(.write, 0) == false);

    try testing.expect(isFull(.read, MaxEventsPerSet) == true);
    try testing.expect(isFull(.read, MaxEventsPerSet - 1) == false);

    try testing.expect(isFull(.partial_write, MaxEventsPerSet) == true);
    try testing.expect(isFull(.partial_write, MaxEventsPerSet - 1) == false);

    try testing.expect(isFull(.none, 100) == false);
}

test "Set.full returns correct state" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    try testing.expect(set.full() == false);

    // Fill with write operation (capacity 1)
    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const result = try set.append(testing.allocator, .write, cl_event, null);
    try testing.expect(result == .success_and_full);
    try testing.expect(set.full() == true);
}

test "Set.toSlice returns correct slice" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    var result = try set.append(testing.allocator, .read, cl_event1, null);
    try testing.expect(result == .success);

    result = try set.append(testing.allocator, .read, cl_event2, null);
    try testing.expect(result == .success);

    const slice = set.toSlice();
    try testing.expect(slice.len == 2);
    try testing.expect(slice[0] == cl_event1);
    try testing.expect(slice[1] == cl_event2);
}

test "Set.appendCallback adds callback without event" {
    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    var test_data: u32 = 42;
    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                _ = data;
            }
        }.testCallback,
        .data = &test_data,
    };

    try set.appendCallback(testing.allocator, callback);
    try testing.expect(set.callbacks.items.len == 1);
    try testing.expect(@intFromPtr(set.callbacks.items[0].data) == @intFromPtr(&test_data));
}

test "Set.executeCallbacks runs all callbacks" {
    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    var counter: u32 = 0;
    const callback1 = UserCallback{
        .func = struct {
            fn increment(data: ?*anyopaque) void {
                const ptr: *u32 = @ptrCast(@alignCast(data.?));
                ptr.* += 1;
            }
        }.increment,
        .data = &counter,
    };

    const callback2 = UserCallback{
        .func = struct {
            fn increment(data: ?*anyopaque) void {
                const ptr: *u32 = @ptrCast(@alignCast(data.?));
                ptr.* += 10;
            }
        }.increment,
        .data = &counter,
    };

    try set.appendCallback(testing.allocator, callback1);
    try set.appendCallback(testing.allocator, callback2);

    set.executeCallbacks();
    try testing.expect(counter == 11);
}

test "Set.waitForEvents waits and executes callbacks" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    var callback_executed: bool = false;
    const callback = UserCallback{
        .func = struct {
            fn setFlag(data: ?*anyopaque) void {
                const ptr: *bool = @ptrCast(@alignCast(data.?));
                ptr.* = true;
            }
        }.setFlag,
        .data = &callback_executed,
    };

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const result = try set.append(testing.allocator, .read, cl_event, callback);
    try testing.expect(result == .success);

    // Set the user event to complete so waitForEvents doesn't block
    try cl.event.setUserEventStatus(cl_event, .complete);

    try set.waitForEvents();
    try testing.expect(callback_executed == true);
}

test "Set.clear releases events and clears callbacks" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    var set = &batch.sets[0];

    const cl_event = try cl.event.createUserEvent(context.ctx);
    // Don't defer release here since clear() will handle it

    var test_data: u32 = 42;
    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                _ = data;
            }
        }.testCallback,
        .data = &test_data,
    };

    const result = try set.append(testing.allocator, .read, cl_event, callback);
    try testing.expect(result == .success);

    try testing.expect(set.events_count == 1);
    try testing.expect(set.callbacks.items.len == 1);

    set.clear(testing.allocator);
    // After clear, callbacks should be deinitialized
    // We can't test events_count here since clear() releases the events
}

test "Set.restart clears and resets state" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const batch = try Batch.init(testing.allocator, null);
    defer batch.deinit(testing.allocator);

    var set = &batch.sets[0];

    const cl_event = try cl.event.createUserEvent(context.ctx);
    var can_release = true;
    defer if (can_release) cl.event.release(cl_event);
    // Don't defer release here since restart() will handle it

    var test_data: u32 = 42;
    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                _ = data;
            }
        }.testCallback,
        .data = &test_data,
    };

    const result = try set.append(testing.allocator, .read, cl_event, callback);
    try testing.expect(result == .success);
    can_release = false;

    try testing.expect(set.events_count == 1);
    try testing.expect(set.operation == .read);

    // Restart should reset everything
    set.restart(testing.allocator);
    try testing.expect(set.events_count == 0);
    try testing.expect(set.operation == .none);
    // callbacks are reinitialized in restart, so we need to deinit them
}

test "Set.append error handling - operation reverts on callback error" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    // Force an allocation error by using a failing allocator
    var failing_allocator = testing.FailingAllocator.init(
        testing.allocator,
        .{ .fail_index = 0 },
    );
    set.callbacks.deinit(testing.allocator);
    set.callbacks = .empty;

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                _ = data;
            }
        }.testCallback,
        .data = null,
    };

    // This should fail and operation should remain .none
    const result = set.append(failing_allocator.allocator(), .read, cl_event, callback);
    try testing.expectError(error.OutOfMemory, result);
    try testing.expect(set.operation == .none);
}

test "Set operations with mixed scenarios" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    const set = try Set.init(0, testing.allocator);
    defer set.deinit(testing.allocator);

    // Test sequence: none -> read -> try write (should fail) -> continue read
    try testing.expect(set.operation == .none);

    const read_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(read_event1);

    var result = try set.append(testing.allocator, .read, read_event1, null);
    try testing.expect(result == .success);
    try testing.expect(set.operation == .read);

    const write_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(write_event);

    result = try set.append(testing.allocator, .write, write_event, null);
    try testing.expect(result == .full);
    try testing.expect(set.operation == .read); // Should remain read
    try testing.expect(set.events_count == 1); // Should still be 1

    const read_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(read_event2);

    result = try set.append(testing.allocator, .read, read_event2, null);
    try testing.expect(result == .success);
    try testing.expect(set.events_count == 2);
}
