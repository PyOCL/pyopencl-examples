
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE |
                          CLK_FILTER_NEAREST;

__kernel void to_blur(int aMaskSize,
                      __read_only image2d_t srcImg,
                      __write_only image2d_t dstImg) {

  int global_id = get_global_id(0);
  int offset = aMaskSize / 2;
  int sum_r = 0;
  int sum_g = 0;
  int sum_b = 0;
  int width = get_image_width(srcImg);
  int height = get_image_height(srcImg); 
  int row_n = global_id / width;
  int col_n = global_id % width;
  uint4 color;
  int2 coord;
  int count = 0;

  for(int y=-offset; y <= offset; y++) {
    for(int x=-offset; x <= offset; x++) {
      coord = (int2)(col_n + x, row_n + y);
      if (coord.x < 0 || coord.x >= width || coord.y < 0 || coord.y >= height) {
        continue;
      }
      color = read_imageui(srcImg, sampler, coord);
      sum_r += color.x;
      sum_g += color.y;
      sum_b += color.z;
      count += 1;
    }
  }
  color.x = sum_r / count;
  color.y = sum_g / count;
  color.z = sum_b / count;
  coord = (int2)(col_n, row_n);
  write_imageui(dstImg, coord, color);
}
