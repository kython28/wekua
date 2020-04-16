__kernel void cut(__global double *a, __global double *b, unsigned int x, unsigned int y, unsigned int w, unsigned int h){
  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  a[i*w+j] = b[(i+y)*h+j+x];
}
