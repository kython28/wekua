#include "wekua.h"

__kernel void rmsprop_kernel(
    __global wk *const restrict x,
    __constant const wk *const restrict gradients,
    __global wk *const restrict gradients_history,

    const ulong row_pitch,
    const ulong slice_pitch,

    const wks lr,
    const wk gamma

#if WK_COMPLEX
    , const wks lri,
    const wks gammai
#endif
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * row_pitch + j * slice_pitch + (k << 1);

    wk gradient = gradients[index];
    wk gradient_i = gradients[index + 1];

    wk gradient_history = gradients_history[index];
    wk gradient_history_i = gradients_history[index + 1];

    COMPLEX_MUL_K(wk)
    COMPLEX_MUL(gradient_history, gradient_history_i, gamma, gamma_i)

    const wk gamma2 = 1 - gamma;
    const wk gamma2_i = gammai;

    wk squared_gradient = gradient;
    wk squared_gradient_i = gradient_i;

    COMPLEX_MUL(squared_gradient, squared_gradient_i, gradient, gradient_i)
    COMPLEX_MUL(squared_gradient, squared_gradient_i, gamma2, gamma2_i)

    gradient_history += squared_gradient;
    gradient_history_i += squared_gradient_i;

    const wk r2 = gradient_history * gradient_history + gradient_history_i * gradient_history_i;
    const wk r = sqrt(r2);
    const wk theta = atan2(gradient_history_i, gradient_history);

    const wk root_r = sqrt(r);

    const wk root_real = root_r * cos(theta/2) + FLT_EPSILON;
    const wk root_imag = root_r * sin(theta/2);

    const wk denominator = root_real / r2;
    const wk denominator_i = -root_imag / r2;

    COMPLEX_MUL(gradient, gradient_i, denominator, denominator_i)
    COMPLEX_MUL(gradient, gradient_i, lr, lri)

    x[index] -= gradient;
    x[index + 1] -= gradient_i;

    gradients_history[index] = gradient_history;
    gradients_history[index + 1] = gradient_history_i;
#else
    const ulong index = i * row_pitch + j * slice_pitch + k;

    const wk gradient = gradients[index];
    const wk gradient_history = gamma * gradients_history[index] + (1 - gamma) * gradient * gradient;

    x[index] -= lr * gradient / (sqrt(gradient_history) + FLT_EPSILON);
    gradients_history[index] = gradient_history;
#endif
}
