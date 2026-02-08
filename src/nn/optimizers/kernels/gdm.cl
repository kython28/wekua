#include "wekua.h"

__kernel void gdm_kernel(
    __global wk *const restrict x,
    __constant const wk *const restrict gradients,
    __global wk *const restrict velocities,
    const wk lr,
    const wk beta
) {
    const ulong index = get_global_id(0);

#if WK_COMPLEX
    wk old_velocity = velocities[index];
    wk gradient = gradients[index];

    wk v_scaled;
    COMPLEX_MUL(old_velocity, beta, v_scaled);

    wk g_scaled;
    COMPLEX_MUL(gradient, lr, g_scaled);

    wk new_velocity = (wk){ v_scaled.real + g_scaled.real, v_scaled.imag + g_scaled.imag };

    wk xval = x[index];
    x[index] = (wk){ xval.real - new_velocity.real, xval.imag - new_velocity.imag };
    velocities[index] = new_velocity;
#else
    const wk new_velocity = beta * velocities[index] + lr * gradients[index];

    x[index] -= new_velocity;
    velocities[index] = new_velocity;
#endif
}
