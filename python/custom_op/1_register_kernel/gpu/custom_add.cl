#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void custom_add_kernel(
    __global const half* inp0, 
    __global const half* inp1,
    __global half* outp) {
    int b = get_global_id(0);
    int f = get_global_id(1);
    int y = get_global_id(2);
    // 20, 5, 10
    int id = b * 5 * 10 + f * 10 + y;

    const float my_bias = 0.1f;

    outp[id] = inp0[id] + inp1[0];
    // outp[id] = inp0[id] + 0.1f;

    // printf("b=%d, ", b);
    // if (id == 0) {
    //     printf("inp0[0]=%.2f, inp1[0]=%.2f, b=%d, f=%d, y=%d, cur_id=%d, my_bias=%.2f, outp[0]=%.2f\n", 
    //         (float)inp0[0], (float)inp1[0], b, f, y, id, my_bias, (float)outp[0]);
    // }
}