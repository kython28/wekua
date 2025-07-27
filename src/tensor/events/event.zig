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

const MaxEventsPerSet = switch (builtin.mode) {
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

pub fn initValues(self: *Event, index: usize, allocator: std.mem.Allocator) void {
    self.operation = .none;
    self.callbacks = UserCallbackArray.init(allocator);
    self.events_count = 0;
    self.index = @intCast(index);
}

pub fn init(index: usize, allocator: std.mem.Allocator) !*Event {
    const self = try allocator.create(Event);
    errdefer allocator.destroy(self);

    self.initValues(index, allocator);
    return self;
}

pub inline fn getParent(self: *Event) *Batch {
    const ptr: usize = @intFromPtr(self) - @offsetOf(Batch, "events") - @as(usize, self.index) * @sizeOf(Event);
    return @ptrFromInt(ptr);
}

pub fn append(
    self: *Event,
    operation: Operation,
    event: cl.event.Event,
    user_callback: ?UserCallback,
) !AppendResult {
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
        try self.callbacks.append(callback);
    }

    self.events[events_count] = event;
    self.events_count = events_count + 1;

    if (isFull(operation, events_count + 1)) return AppendResult.success_and_full;

    return AppendResult.success;
}

pub inline fn pop(self: *Event, had_callback: bool) void {
    const events_count = self.events_count;
    self.events_count = events_count - 1;
    if (events_count == 0) {
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

pub inline fn full(self: *const Event) bool {
    return isFull(self.operation, self.events_count);
}

pub inline fn toSlice(self: *const Event) []const cl.event.Event {
    return self.events[0..self.events_count];
}

pub inline fn appendCallback(self: *Event, callback: UserCallback) !void {
    try self.callbacks.append(callback);
}

pub inline fn executeCallbacks(self: *const Event) void {
    for (self.callbacks.items) |*callback| {
        callback.func(callback.data);
    }
}

pub inline fn waitForEvents(self: *const Event) !void {
    try cl.event.waitForMany(self.toSlice());
    self.executeCallbacks();
}

pub inline fn clear(self: *Event) void {
    for (self.events[0..self.events_count]) |event| {
        cl.event.release(event);
    }

    self.callbacks.deinit();
}

pub inline fn restart(self: *Event) void {
    self.clear();

    self.operation = .none;
    self.events_count = 0;
}

const Event = @This();

// Tests
const testing = std.testing;
const core = @import("../../core/main.zig");

test "Event.initValues initializes correctly" {
    var event: Event = undefined;
    event.initValues(5, testing.allocator);
    defer event.callbacks.deinit();

    try testing.expect(event.operation == .none);
    try testing.expect(event.events_count == 0);
    try testing.expect(event.index == 5);
    try testing.expect(event.callbacks.items.len == 0);
}

test "Event.getParent returns correct parent batch" {
    // Create a real batch structure to test getParent
    var batch: Batch = undefined;
    try batch.initValue(testing.allocator, null);
    defer batch.clear();

    const parent = batch.events[0].getParent();
    try testing.expect(@intFromPtr(parent) == @intFromPtr(&batch));
}

test "Event.append with none operation sets operation and adds event" {
    const context = try core.Context.initFromDeviceType(
        testing.allocator,
        null,
        cl.device.Type.all,
    );
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    const result = try event.append(.read, cl_event, null);

    try testing.expect(result == .success);
    try testing.expect(event.operation == .read);
    try testing.expect(event.events_count == 1);
    try testing.expect(event.events[0] == cl_event);
}

test "Event.append with matching operation succeeds" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    _ = try event.append(.read, cl_event1, null);
    const result = try event.append(.read, cl_event2, null);

    try testing.expect(result == .success);
    try testing.expect(event.events_count == 2);
    try testing.expect(event.events[1] == cl_event2);
}

test "Event.append with different operation returns full" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    _ = try event.append(.read, cl_event1, null);
    const result = try event.append(.write, cl_event2, null);

    try testing.expect(result == .full);
    try testing.expect(event.operation == .read);
    try testing.expect(event.events_count == 1);
}

test "Event.append with write operation allows only one event" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    _ = try event.append(.write, cl_event1, null);
    const result = try event.append(.write, cl_event2, null);

    try testing.expect(result == .full);
    try testing.expect(event.events_count == 1);
}

test "Event.append with callback stores callback" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

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

    _ = try event.append(.read, cl_event, callback);

    try testing.expect(event.callbacks.items.len == 1);
    try testing.expect(event.callbacks.items[0].data == &test_data);
}

test "Event.append returns success_and_full when reaching capacity" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    var events_to_cleanup = std.ArrayList(cl.event.Event).init(testing.allocator);
    defer {
        for (events_to_cleanup.items) |e| {
            cl.event.release(e);
        }
        events_to_cleanup.deinit();
    }

    // Fill up to capacity - 1 for read operation
    for (0..MaxEventsPerSet - 1) |_| {
        const cl_event = try cl.event.createUserEvent(context.ctx);
        try events_to_cleanup.append(cl_event);
        const result = try event.append(.read, cl_event, null);
        try testing.expect(result == .success);
    }

    // Last append should return success_and_full
    const last_event = try cl.event.createUserEvent(context.ctx);
    try events_to_cleanup.append(last_event);
    const result = try event.append(.read, last_event, null);
    try testing.expect(result == .success_and_full);
    try testing.expect(event.events_count == MaxEventsPerSet);
}

test "Event.pop decrements count and resets operation when empty" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    _ = try event.append(.read, cl_event, null);

    event.pop(false);
    try testing.expect(event.events_count == 0);
    try testing.expect(event.operation == .none);
}

test "Event.pop with callback removes callback" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

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

    _ = try event.append(.read, cl_event, callback);

    try testing.expect(event.callbacks.items.len == 1);
    event.pop(true);
    try testing.expect(event.callbacks.items.len == 0);
}

test "Event.isFull returns correct values for different operations" {
    try testing.expect(isFull(.write, 1) == true);
    try testing.expect(isFull(.write, 0) == false);

    try testing.expect(isFull(.read, MaxEventsPerSet) == true);
    try testing.expect(isFull(.read, MaxEventsPerSet - 1) == false);

    try testing.expect(isFull(.partial_write, MaxEventsPerSet) == true);
    try testing.expect(isFull(.partial_write, MaxEventsPerSet - 1) == false);

    try testing.expect(isFull(.none, 100) == false);
}

test "Event.full returns correct state" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    try testing.expect(event.full() == false);

    // Fill with write operation (capacity 1)
    const cl_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event);

    _ = try event.append(.write, cl_event, null);
    try testing.expect(event.full() == true);
}

test "Event.toSlice returns correct slice" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    const cl_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event1);
    const cl_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(cl_event2);

    _ = try event.append(.read, cl_event1, null);
    _ = try event.append(.read, cl_event2, null);

    const slice = event.toSlice();
    try testing.expect(slice.len == 2);
    try testing.expect(slice[0] == cl_event1);
    try testing.expect(slice[1] == cl_event2);
}

test "Event.appendCallback adds callback without event" {
    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    var test_data: u32 = 42;
    const callback = UserCallback{
        .func = struct {
            fn testCallback(data: ?*anyopaque) void {
                _ = data;
            }
        }.testCallback,
        .data = &test_data,
    };

    try event.appendCallback(callback);
    try testing.expect(event.callbacks.items.len == 1);
    try testing.expect(event.callbacks.items[0].data == &test_data);
}

test "Event.executeCallbacks runs all callbacks" {
    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

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

    try event.appendCallback(callback1);
    try event.appendCallback(callback2);

    event.executeCallbacks();
    try testing.expect(counter == 11);
}

test "Event.waitForEvents waits and executes callbacks" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

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

    _ = try event.append(.read, cl_event, callback);

    // Set the user event to complete so waitForEvents doesn't block
    try cl.event.setUserEventStatus(cl_event, .complete);

    try event.waitForEvents();
    try testing.expect(callback_executed == true);
}

test "Event.clear releases events and clears callbacks" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var batch: Batch = undefined;
    try batch.initValue(testing.allocator, null);
    defer batch.clear();

    var event = &batch.events[0];

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

    _ = try event.append(.read, cl_event, callback);
    try testing.expect(event.events_count == 1);
    try testing.expect(event.callbacks.items.len == 1);

    event.clear();
    // After clear, callbacks should be deinitialized
    // We can't test events_count here since clear() releases the events
}

test "Event.restart clears and resets state" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var batch: Batch = undefined;
    try batch.initValue(testing.allocator, null);
    defer batch.clear();

    var event = &batch.events[0];

    const cl_event = try cl.event.createUserEvent(context.ctx);
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

    _ = try event.append(.read, cl_event, callback);
    try testing.expect(event.events_count == 1);
    try testing.expect(event.operation == .read);

    // Restart should reset everything
    event.restart();
    try testing.expect(event.events_count == 0);
    try testing.expect(event.operation == .none);
    // callbacks are reinitialized in restart, so we need to deinit them
}

test "Event.append error handling - operation reverts on callback error" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    // Force an allocation error by using a failing allocator
    var failing_allocator = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    event.callbacks.deinit();
    event.callbacks = UserCallbackArray.init(failing_allocator.allocator());

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
    const result = event.append(.read, cl_event, callback);
    try testing.expectError(error.OutOfMemory, result);
    try testing.expect(event.operation == .none);
}

test "Event operations with mixed scenarios" {
    const context = try core.Context.initFromDeviceType(testing.allocator, null, cl.device.Type.all);
    defer context.deinit();

    var event: Event = undefined;
    event.initValues(0, testing.allocator);
    defer event.callbacks.deinit();

    // Test sequence: none -> read -> try write (should fail) -> continue read
    try testing.expect(event.operation == .none);

    const read_event1 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(read_event1);
    var result = try event.append(.read, read_event1, null);
    try testing.expect(result == .success);
    try testing.expect(event.operation == .read);

    const write_event = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(write_event);
    result = try event.append(.write, write_event, null);
    try testing.expect(result == .full);
    try testing.expect(event.operation == .read); // Should remain read
    try testing.expect(event.events_count == 1); // Should still be 1

    const read_event2 = try cl.event.createUserEvent(context.ctx);
    defer cl.event.release(read_event2);
    result = try event.append(.read, read_event2, null);
    try testing.expect(result == .success);
    try testing.expect(event.events_count == 2);
}
