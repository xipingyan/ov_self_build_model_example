#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void custom_eltwise_kernel(
    __global const INPUT0_TYPE* inp0, 
    __global const INPUT1_TYPE* inp1,
    __global const INPUT2_TYPE* inp2,
    __global OUTPUT0_TYPE* outp) {
    int b = get_global_id(0);
    int f = get_global_id(1);
    int y = get_global_id(2);
    #if INPUT0_DIMS_SIZE == 4
        const uint x = 0;
    #endif

    // 20, 5, 10
    // int id = b * 5 * 10 + f * 10 + y;
    const unsigned id = b*INPUT0_DIMS[1]*INPUT0_DIMS[2]*INPUT0_DIMS[3] + f*INPUT0_DIMS[2]*INPUT0_DIMS[3] + y*INPUT0_DIMS[3] + x;

    outp[id] = inp0[id] * inp1[0] + inp2[0];

    // if (id == 0) {
    //     printf("--> kernel: custom_eltwise_kernel is called.\n");
    // }
}