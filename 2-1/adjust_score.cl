__kernel void adjust_score(__global int4* values, __global int4* final_data) {
    int global_id = get_global_id(0);
    final_data[global_id] = convert_int4(sqrt(convert_float4(values[global_id])) * 10);
}
