typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} Pixel;

__kernel void histogram(__global Pixel* pixels, unsigned long max_size, volatile __global unsigned int* result)
{
  unsigned int gid = get_global_id(0);
  unsigned int lid = get_local_id(0);
  unsigned int local_size = get_local_size(0);
  unsigned int mod = 768 % local_size;
  unsigned int batch_size = (mod == 0) ? 768 / local_size : 768 / local_size + 1;
  unsigned i = batch_size * lid;
  unsigned until = i + batch_size;

  volatile __local unsigned int local_buffers[768];

  for (; i < until && i < 768; i++) {
    local_buffers[i] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (gid < max_size) {
    atomic_inc(local_buffers + pixels[gid].red);
    atomic_inc(local_buffers + pixels[gid].green + 256);
    atomic_inc(local_buffers + pixels[gid].blue + 512);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = batch_size * lid ; i < until && i < 768; i++) {
    atomic_add(result + i, local_buffers[i]);
  }
}
