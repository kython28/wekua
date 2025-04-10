#include "wekua.h"

__kernel void gdm_kernel(
    __global wk *const restrict x,
    __constant const wk *const restrict gradients,
    __global wk *const restrict velocities,

    const ulong row_pitch,
    const ulong slice_pitch,

    const wks lr,
    const wks beta

#if WK_COMPLEX
    , const wks lri,
    const wks betai
#endif
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);


#if WK_COMPLEX
    const ulong index = i * row_pitch + j * slice_pitch + (k << 1);

    wk old_velocity = velocities[index];
    wk old_velocity_i = velocities[index + 1];

    wk gradient = gradients[index];
    wk gradient_i = gradients[index + 1];

    COMPLEX_MUL_K(wk)

    COMPLEX_MUL(old_velocity, old_velocity_i, beta, betai)
    COMPLEX_MUL(gradient, gradient_i, lr, lri)

    old_velocity += gradient;
    old_velocity_i += gradient_i;

    x[index] -= old_velocity;
    x[index + 1] -= old_velocity_i;

    velocities[index] = old_velocity;
    velocities[index + 1] = old_velocity_i;
#else
    const ulong index = i * row_pitch + j * slice_pitch + k;
    const wk new_velocity = beta * velocities[index] + lr * gradients[index];

    x[index] -= new_velocity;
    velocities[index] = new_velocity;
#endif
}
