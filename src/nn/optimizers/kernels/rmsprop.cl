#include "wekua.h"

__kernel void rmsprop_kernel(
    __global wk *const restrict x,
    __global const wk *const restrict gradients,
    __global wk *const restrict gradients_history,
    const wk lr,
    const wk gamma
) {
    const ulong index = get_global_id(0);

#if WK_COMPLEX
    wk gradient = gradients[index];
    wk gh = gradients_history[index];

    wk gradient_history;
    COMPLEX_MUL(gh, gamma, gradient_history);

    wk gamma2 = (wk){ (wks)1 - gamma.real, -gamma.imag };

    wk squared_gradient;
    COMPLEX_MUL(gradient, gradient, squared_gradient);

    wk sg_scaled;
    COMPLEX_MUL(squared_gradient, gamma2, sg_scaled);

    gradient_history = (wk){ gradient_history.real + sg_scaled.real, gradient_history.imag + sg_scaled.imag };

    const wks r2 = gradient_history.real * gradient_history.real + gradient_history.imag * gradient_history.imag;
    const wks r = sqrt(r2);
    const wks theta = atan2(gradient_history.imag, gradient_history.real);

    const wks root_r = sqrt(r);

    const wks root_real = root_r * cos(theta/(wks)2) + FLT_EPSILON;
    const wks root_imag = root_r * sin(theta/(wks)2);

    wk denominator = (wk){ root_real / r2, -root_imag / r2 };

    wk tmp;
    COMPLEX_MUL(gradient, denominator, tmp);

    wk result;
    COMPLEX_MUL(tmp, lr, result);

    wk xval = x[index];
    x[index] = (wk){ xval.real - result.real, xval.imag - result.imag };

    gradients_history[index] = gradient_history;
#else
    const wk gradient = gradients[index];
    const wk gradient_history = gamma * gradients_history[index] + ((wks)1 - gamma) * gradient * gradient;

    x[index] -= lr * gradient / (sqrt(gradient_history) + FLT_EPSILON);
    gradients_history[index] = gradient_history;
#endif
}
