__kernel void do_clustering(int num_of_clusters,
                            int num_of_points,
                            __global float* centers_x,
                            __global float* centers_y,
                            __global float* points_x,
                            __global float* points_y,
                            __global int* point_cluster_ids)
{
  int global_id = get_global_id(0);
  if (global_id >= num_of_points) {
    return;
  }

  int point_cluster_id = point_cluster_ids[global_id];
  float x = points_x[global_id];
  float y = points_y[global_id];
  // printf("[do_clustering][%d] (%f, %f), in cluster : %d \n", global_id, x, y, point_cluster_id);
  float min = FLT_MAX;
  float dist = 0;
  int min_id = point_cluster_id;
  float c_x, c_y;
  for (int i = 0; i < num_of_clusters; i++) {
    c_x = centers_x[i];
    c_y = centers_y[i];
    dist = distance((float2) (x, y), (float2) (c_x, c_y));
    // printf("[do_c][%d] compare to cid(%d)(%f, %f), dist = %f \n",
    //   global_id, i, c_x, c_y, dist);
    if (dist < min) {
      min = dist;
      min_id = i;
    }
  }
  // printf("[do_clustering][%d] clustering, original cid = %d, new cid = %d \n",
  //   global_id, point_cluster_id, min_id);
  point_cluster_ids[global_id] = min_id;
}

__kernel void calc_centroid(int num_of_clusters,
                            int num_of_points,
                            __global float* centers_x,
                            __global float* centers_y,
                            __global float* points_x,
                            __global float* points_y,
                            __global int* point_cluster_ids)
{
  // 分 N 群, 每一群的重心計算為一個 kernel task.
  int global_id = get_global_id(0);
  if (global_id >= num_of_clusters) {
    return;
  }
  float sum_x = 0;
  float sum_y = 0;
  int count = 0;
  for (int i = 0; i < num_of_points; i++) {
    if (point_cluster_ids[i] == global_id) {
      sum_x += points_x[i];
      sum_y += points_y[i];
      count += 1;
    }
  }
  float new_x = sum_x / count;
  float new_y = sum_y / count;
  // printf("[cacl_centroid][%d] center = (%f, %f) \n", global_id, new_x, new_y);
  centers_x[global_id] = new_x;
  centers_y[global_id] = new_y;
}