
typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} Pixel;

__kernel void to_gray(int aWidth,
                      int aHeight,
                      __global Pixel* aBufferIn,
                      __global Pixel* aBufferOut) {
  int global_id = get_global_id(0);
  int gray = (aBufferIn[global_id].red + aBufferIn[global_id].green + aBufferIn[global_id].blue) / 3;
  aBufferOut[global_id].red = gray;
  aBufferOut[global_id].green = gray;
  aBufferOut[global_id].blue = gray;
}
