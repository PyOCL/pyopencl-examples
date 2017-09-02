#define PIXELS_PER_ITEM 256

typedef struct {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
} Pixel;

__kernel void histogram(__global Pixel* pixels, unsigned long max_size, volatile __global unsigned int* result)
{
  unsigned int gid = get_global_id(0);
  unsigned int lid = get_local_id(0);
  unsigned int local_size = get_local_size(0);
  unsigned int mod = 768 % local_size;
  unsigned int batch_size = (mod == 0) ? 768 / local_size : 768 / local_size + 1;
  unsigned int i = batch_size * lid;
  unsigned int until = i + batch_size;
  unsigned int pixel_start_index = gid * PIXELS_PER_ITEM;
  unsigned int pixel_end_index = pixel_start_index + PIXELS_PER_ITEM;

  volatile __local unsigned int local_buffers[768];

  for (; i < until && i < 768; i++) {
    local_buffers[i] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = pixel_start_index; i < pixel_end_index && i < max_size; i++) {
    atomic_inc(local_buffers + pixels[i].red);
    atomic_inc(local_buffers + pixels[i].green + 256);
    atomic_inc(local_buffers + pixels[i].blue + 512);
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (i = batch_size * lid ; i < until && i < 768; i++) {
    atomic_add(result + i, local_buffers[i]);
  }
}
