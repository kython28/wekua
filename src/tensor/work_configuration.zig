const std = @import("std");

const utils = @import("utils");

const core = @import("core");
const CommandQueue = core.CommandQueue;


pub const GemmAlgorithm = enum(u8) {
    generic = 0,
    @"4x4" = 1,
    @"8x8" = 2,
    @"16x16" = 3,
    @"32x32" = 4,
};

global_work_items: [3]u64,
global_work_items_without_vectors: [3]u64,

local_work_items_1d: []u64,
local_work_items_for_vectors_1d: []u64,

local_work_items: [][3]u64,
local_work_items_without_vectors: [][3]u64,

gemm_algorithm_per_device: []GemmAlgorithm,
global_work_items_gemm_generic: [2]u64,
local_work_items_gemm_generic: [][2]u64,

global_work_items_gemm_4x4: [][2]u64,
local_work_items_gemm_4x4: [][2]u64,

global_work_items_gemm_8x8: [][2]u64,
local_work_items_gemm_8x8: [][2]u64,

global_work_items_gemm_16x16: [][2]u64,
local_work_items_gemm_16x16: [][2]u64,

global_work_items_gemm_32x32: [][2]u64,
local_work_items_gemm_32x32: [][2]u64,

pub fn init(
    self: *WorkConfiguration,
    comptime T: type,
    arena_allocator: std.mem.Allocator,
    command_queues: []const CommandQueue,
    depth: u64,
    penultimate_size: u64,
    padded_penultimate_size: u64,
    number_of_elements: u64,
    number_of_vectors: u64,
    last_size: u64,
    vl_shape: []const u64,
) error{OutOfMemory}!void {
    const local_work_items_1d = try arena_allocator.alloc(u64, command_queues.len);
    const lobal_work_items_for_vectors_1d = try arena_allocator.alloc(u64, command_queues.len);
    const local_work_items = try arena_allocator.alloc([3]u64, command_queues.len);
    const local_work_items_without_vectors = try arena_allocator.alloc([3]u64, command_queues.len);

    self.local_work_items_1d = local_work_items_1d;
    self.local_work_items_for_vectors_1d = lobal_work_items_for_vectors_1d;
    self.local_work_items = local_work_items;
    self.local_work_items_without_vectors = local_work_items_without_vectors;

    const global_work_items: []u64 = &self.global_work_items;
    const global_work_items_without_vectors: []u64 = &self.global_work_items_without_vectors;

    global_work_items[0] = depth;
    global_work_items_without_vectors[0] = depth;

    global_work_items[1] = penultimate_size;
    global_work_items_without_vectors[1] = penultimate_size;

    global_work_items_without_vectors[2] = last_size;
    global_work_items[2] = vl_shape[vl_shape.len - 1];

    for (
        command_queues,
        local_work_items_1d,
        lobal_work_items_for_vectors_1d,
        local_work_items,
        local_work_items_without_vectors,
    ) |cmd, *wa, *wv, *wmv, *wm| {
        utils.calculateWorkItems(
            @as([*]const u64, @ptrCast(&number_of_elements))[0..1],
            @as([*]u64, @ptrCast(wa))[0..1],
            cmd.max_work_group_size,
        );

        utils.calculateWorkItems(
            @as([*]const u64, @ptrCast(&number_of_vectors))[0..1],
            @as([*]u64, @ptrCast(wv))[0..1],
            cmd.max_work_group_size,
        );

        utils.calculateWorkItems(global_work_items, wmv, cmd.max_work_group_size);
        utils.calculateWorkItems(global_work_items_without_vectors, wm, cmd.max_work_group_size);
    }

    try self.initGemm(T, arena_allocator, command_queues, padded_penultimate_size, last_size);
}

fn initGemm(
    self: *WorkConfiguration,
    comptime T: type,
    arena_allocator: std.mem.Allocator,
    command_queues: []const CommandQueue,
    padded_penultimate_size: u64,
    last_size: u64,
) error{OutOfMemory}!void {
    const local_work_items_gemm = try arena_allocator.alloc([2]u64, command_queues.len);
    self.local_work_items_gemm_generic = local_work_items_gemm;

    const gemm_algorithm_per_device = try arena_allocator.alloc(GemmAlgorithm, command_queues.len);
    @memset(gemm_algorithm_per_device, GemmAlgorithm.generic);
    self.gemm_algorithm_per_device = gemm_algorithm_per_device;

    const gwi_h = padded_penultimate_size;
    const gwi_w = (last_size + (last_size % 2));

    self.global_work_items_gemm_generic[0] = gwi_h / 2;
    self.global_work_items_gemm_generic[1] = gwi_w / 2;

    for (command_queues, local_work_items_gemm) |cmd, *lw_gemm| {
        utils.calculateWorkItems(
            &self.global_work_items_gemm_generic,
            lw_gemm,
            cmd.max_work_group_size,
        );
    }

    var max_block_length: u8 = 0;
    comptime var block_length = 4;
    inline while (block_length < 64) : (block_length *= 2) {
        const algorithm_name = std.fmt.comptimePrint("{0}x{0}", .{block_length});
        const global_field_name = std.fmt.comptimePrint("global_work_items_gemm_{s}", .{algorithm_name});
        const local_field_name = std.fmt.comptimePrint("local_work_items_gemm_{s}", .{algorithm_name});
        if ((gwi_h % block_length == 0) and (gwi_w % block_length == 0)) {
            @field(self, global_field_name) = try arena_allocator.alloc([2]u64, command_queues.len);
            @field(self, local_field_name) = try arena_allocator.alloc([2]u64, command_queues.len);
            max_block_length = block_length;
        } else {
            @field(self, global_field_name) = &.{};
            @field(self, local_field_name) = &.{};
        }
    }

    for (command_queues, 0..) |cmd, i| {
        const type_index = core.types.getTypeId(T);
        comptime var block_length2 = 4;
        var algorithm: GemmAlgorithm = .generic;
        inline while (block_length2 < 64) : (block_length2 *= 2) {
            const block_size = cmd.vector_widths[type_index] * block_length2 * @sizeOf(T);
            const blocks_fit_in_local_mem = switch (cmd.local_mem_type) {
                .local => (block_size * 2 * 4 <= cmd.local_mem_size),
                .global => ((block_size * block_length2) <= 16 * 1024),
            };

            if (block_length2 <= max_block_length and blocks_fit_in_local_mem) {
                const algorithm_name = std.fmt.comptimePrint("{0}x{0}", .{block_length2});
                const g_field_name = std.fmt.comptimePrint("global_work_items_gemm_{s}", .{algorithm_name});
                const l_field_name = std.fmt.comptimePrint("local_work_items_gemm_{s}", .{algorithm_name});
                const g_values = &@field(self, g_field_name)[i];
                switch (cmd.local_mem_type) {
                    .local => @memcpy(g_values, &self.global_work_items_gemm_generic),
                    .global => {
                        g_values[0] = gwi_h / block_length2;
                        g_values[1] = gwi_w / block_length2;
                    },
                }
                algorithm = @field(GemmAlgorithm, algorithm_name);
                utils.calculateWorkItems(
                    g_values,
                    &@field(self, l_field_name)[i],
                    @min(block_length2 * block_length2, cmd.max_work_group_size),
                );
            }
        }
        self.gemm_algorithm_per_device[i] = algorithm;
    }
}

const WorkConfiguration = @This();
