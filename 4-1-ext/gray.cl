
typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} Pixel;

__kernel void to_gray(int aWidth,
                      int aHeight,
                      __global Pixel* aBufferIn,
                      __global Pixel* aBufferOut) {
  int global_size_0 = get_global_size(0);
  int global_id_0 = get_global_id(0);
  int global_id_1 = get_global_id(1);

  int offset_0 = get_global_offset(0);
  int offset_1 = get_global_offset(1);

  int index_0 = global_id_0 - offset_0;
  int index_1 = global_id_1 - offset_1;
  int index = index_1 * global_size_0 + index_0;

  if (index >= aWidth * aHeight) {
    return;
  }

  int gray = (aBufferIn[index].red + aBufferIn[index].green + aBufferIn[index].blue) / 3;
  aBufferOut[index].red = gray;
  aBufferOut[index].green = gray;
  aBufferOut[index].blue = gray;
}
