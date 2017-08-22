
typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} Pixel;

__kernel void to_gray(int aWidth,
                      int aHeight,
                      __global Pixel* aBufferIn,
                      __global Pixel* aBufferOut) {
  int wd = get_work_dim();
  if (wd == 2) {
    int wgs_x = get_global_size(0);
    int wgs_y = get_global_size(1);
    int lgs_x = get_local_size(0);
    int lgs_y = get_local_size(1);
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    int index = global_id_x + global_id_y * wgs_x;

    int gray = (aBufferIn[index].red + aBufferIn[index].green + aBufferIn[index].blue) / 3;
    aBufferOut[index].red = gray;
    aBufferOut[index].green = gray;
    aBufferOut[index].blue = gray;
  } else {
    int global_id = get_global_id(0);
    int gray = (aBufferIn[global_id].red + aBufferIn[global_id].green + aBufferIn[global_id].blue) / 3;
    aBufferOut[global_id].red = gray;
    aBufferOut[global_id].green = gray;
    aBufferOut[global_id].blue = gray;
  }
}
