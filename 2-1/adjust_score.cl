__kernel void adjust_score(__global int4* values, __global int4* final) {
    int global_id = get_global_id(0);
    final[global_id] = convert_int4(sqrt(convert_float4(values[global_id])) * 10);
}
