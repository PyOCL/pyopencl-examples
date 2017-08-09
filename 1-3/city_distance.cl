void calc_pq_index(int size, int idx, int* p, int* q);

void calc_pq_index(int size, int idx, int* p, int* q) {
    int i;
    int count = idx;
    *p = 0;
    *q = *p + 1;
    for (i = size - 1; i > 0; i--) {
        if (count < i) {
            *q = count + *q;
            break;
        } else {
            count -= i;
            *p = *p + 1;
            *q = *p + 1;
        }
    }
}

__kernel void calc_distance(int size, __global float* x, __global float* y, __global float* final) {
    int global_id = get_global_id(0);
    int p_idx, q_idx;

    calc_pq_index(size, global_id, &p_idx, &q_idx);

    float2 p = (float2)(x[p_idx], y[p_idx]);
    float2 q = (float2)(x[q_idx], y[q_idx]);
    final[global_id] = distance(p, q);
}
