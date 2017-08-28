
__kernel void to_gray(__global uchar* aBufferIn,
                      __global uchar* aBufferOut) {
  int global_id = get_global_id(0);
  int index = global_id * 4;
  uchar r = aBufferIn[index];
  uchar g = aBufferIn[index + 1];
  uchar b = aBufferIn[index + 2];
  uchar gray = (r + g + b) / 3;
  aBufferOut[index] = gray;
  aBufferOut[index + 1] = gray;
  aBufferOut[index + 2] = gray;
  aBufferOut[index + 3] = aBufferIn[index + 3];
}

__kernel void to_gray4(__global uchar4* aBufferIn,
                       __global uchar4* aBufferOut) {
int global_id = get_global_id(0);
uchar gray = (aBufferIn[global_id].x + aBufferIn[global_id].y + aBufferIn[global_id].z) / 3;
aBufferOut[global_id] = (uchar4)(gray, gray, gray, aBufferIn[global_id].w);
}