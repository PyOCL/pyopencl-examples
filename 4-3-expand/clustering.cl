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

  float2 xy = (float2) (points_x[global_id], points_y[global_id]);
  float dist0 = distance(xy, (float2) (centers_x[0], centers_y[0]));
  float dist1 = distance(xy, (float2) (centers_x[1], centers_y[1]));
  float dist2 = distance(xy, (float2) (centers_x[2], centers_y[2]));
  float dist3 = distance(xy, (float2) (centers_x[3], centers_y[3]));
  float dist4 = distance(xy, (float2) (centers_x[4], centers_y[4]));
  float4 check0 = (float4) (dist1, dist2, dist3, dist4);
  float4 check1 = (float4) (dist0, dist2, dist3, dist4);
  float4 check2 = (float4) (dist0, dist1, dist3, dist4);
  float4 check3 = (float4) (dist0, dist2, dist1, dist4);
  float4 check4 = (float4) (dist0, dist2, dist3, dist1);
  int4 less0 = isless((float4) dist0, check0);
  int4 less1 = isless((float4) dist1, check1);
  int4 less2 = isless((float4) dist2, check2);
  int4 less3 = isless((float4) dist3, check3);
  if (all(less0)) {
    point_cluster_ids[global_id] = 0;
  } else if (all(less1)) {
    point_cluster_ids[global_id] = 1;
  } else if (all(less2)) {
    point_cluster_ids[global_id] = 2;
  } else if (all(less3)) {
    point_cluster_ids[global_id] = 3;
  } else {
    point_cluster_ids[global_id] = 4;
  }
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