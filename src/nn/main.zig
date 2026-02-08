pub const activation_module = @import("activation/main.zig");
pub const layer_module = @import("layer/main.zig");
pub const optimizer_module = @import("optimizers/main.zig");
pub const loss_module = @import("loss/main.zig");

test {
    _ = activation_module;
    _ = layer_module;
    _ = optimizer_module;
    _ = loss_module;
}
