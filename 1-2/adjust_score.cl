__kernel void adjust_score(__global int* values, __global int* final) {
    int global_id = get_global_id(0);
    final[global_id] = convert_int(sqrt(convert_float(values[global_id])) * 10);
}
