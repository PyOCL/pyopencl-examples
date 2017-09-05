
__kernel void to_gray(__global uchar* aBufferIn,
                      __global uchar* aBufferOut) {
  int global_id = get_global_id(0);
  int index = global_id * 4;
  uchar r = aBufferIn[index];
  uchar g = aBufferIn[index + 1];
  uchar b = aBufferIn[index + 2];
  uchar gray =  0.299 * r + 0.587 *g + 0.114 * b;
  aBufferOut[index] = gray;
  aBufferOut[index + 1] = gray;
  aBufferOut[index + 2] = gray;
  aBufferOut[index + 3] = aBufferIn[index + 3];
}

__kernel void to_gray4(__global uchar4* aBufferIn,
                       __global uchar4* aBufferOut) {
  int global_id = get_global_id(0);
  uchar4 data = aBufferIn[global_id];
  float4 mask = (float4) (0.299, 0.587, 0.114, 1);
  float4 final = dot(convert_float4(data), mask);
  aBufferOut[global_id] = (uchar4) ((uchar)final.x, (uchar)final.y, (uchar)final.z, (uchar)final.w);
}