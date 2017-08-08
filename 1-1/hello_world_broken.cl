__kernel void hello_world() {
    int global_id = get_global_id(0);
    printf("hello host from kernel #%d\n", global_id);
}
