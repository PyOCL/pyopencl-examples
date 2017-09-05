__kernel void adjust_score(__global int4* values, __global int4* final) {
    int global_id = get_global_id(0);
    // convert int4 to float4 with implicit data type conversion
    float4 float_value = (float4) (values[global_id].x,
                                   values[global_id].y,
                                   values[global_id].z,
                                   values[global_id].w);
    // do calculation
    float4 float_final = sqrt(float_value) * 10;
    // convert float4 to int4 with implicit data type conversion
    final[global_id] = (int4) (float_final.x,
                               float_final.y,
                               float_final.z,
                               float_final.w);
}
