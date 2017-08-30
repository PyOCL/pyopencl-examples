
__kernel void exec_work_item() {
  int global_size_0 = get_global_size(0);
  int global_size_1 = get_global_size(1);
  int global_id_0 = get_global_id(0);
  int global_id_1 = get_global_id(1);

  int local_id_0 = get_local_id(0);
  int local_id_1 = get_local_id(1);

  int offset_0 = get_global_offset(0);
  int offset_1 = get_global_offset(1);

  int index_0 = global_id_0 - offset_0;
  int index_1 = global_id_1 - offset_1;
  int index = index_1 * global_size_0 + index_0;

  if (index >= global_size_0 * global_size_1 ||
      global_id_1 >= global_size_1 || global_id_0 >= global_size_0) {
    return;
  }

  printf("gs(%d, %d), gid(%d, %d), local_id(%d, %d), goid(%d, %d), index = %d \n",
    global_size_0, global_size_1,
    global_id_0, global_id_1,
    local_id_0, local_id_1,
    offset_0, offset_1,
    index);
}
