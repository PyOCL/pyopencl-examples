#define PIXELS_PER_ITEM 256

typedef struct {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
} Pixel;

__kernel void histogram(__global Pixel* pixels, unsigned long max_size, volatile __global unsigned int* result)
{
  unsigned int gid = get_global_id(0);
  unsigned int pixel_start_index = gid * PIXELS_PER_ITEM;
  unsigned int pixel_end_index = pixel_start_index + PIXELS_PER_ITEM;
  unsigned int i;

  for (i = pixel_start_index; i < pixel_end_index && i < max_size; i++) {
    atomic_inc(result + pixels[i].red);
    atomic_inc(result + pixels[i].green + 256);
    atomic_inc(result + pixels[i].blue + 512);
  }
}
