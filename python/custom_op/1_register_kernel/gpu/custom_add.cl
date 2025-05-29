#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void custom_add_kernel(
    __global const float* inp0, 
    __global const float* inp1,
    __global float* outp) {
    int b = get_global_id(0);
    int f = get_global_id(1);
    int y = get_global_id(2);
    // 20, 5, 10
    int id = b*5*10+f*10+y;

    const float my_bias = 0.1f;
    printf("inp0[0]=%f, inp1[0]=%f, b=%d, f=%d, y=%d, cur_id=%d, my_bias=%f\n", inp0[0], inp1[0], b, f, y, id, my_bias);

    // outp[id] = inp0[id] + inp1[0];
    outp[id] = inp0[id] + my_bias;
}