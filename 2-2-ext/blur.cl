
typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} Pixel;

int convert_row_col_2_id(int w, int r, int c) {
  return w * r + c;
}

__kernel void to_blur(int aMaskSize,
                      int aWidth,
                      int aHeight,
                      __global Pixel* aBufferIn,
                      __global Pixel* aBufferOut) {
  int global_id = get_global_id(0);
  int offset = aMaskSize >> 1;
  int sum_r = 0;
  int sum_g = 0;
  int sum_b = 0;
  int row_n = global_id / aWidth;
  int col_n = global_id % aWidth;
  int tmp_id = 0;
  int count = 0;
  for(int y=-offset; y <= offset; y++) {
      for(int x=-offset; x <= offset; x++) {
      tmp_id = convert_row_col_2_id(aWidth, row_n + y, col_n + x);
      if (tmp_id < 0 || tmp_id >= aWidth * aHeight) {
        continue;
      }
      sum_r += aBufferIn[tmp_id].red;
      sum_g += aBufferIn[tmp_id].green;
      sum_b += aBufferIn[tmp_id].blue;
      count += 1;
    }
  }
  aBufferOut[global_id].red = sum_r / count;
  aBufferOut[global_id].green = sum_g / count;
  aBufferOut[global_id].blue = sum_b / count;
}
