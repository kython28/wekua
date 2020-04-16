__kernel void rand(__global double *a, __global long *b){
  unsigned int i = get_global_id(0);
  double RECIP_BPF = 0.00000000000000011102230246251565404236316680908203125;
  a[i] = (b[i] >> 4)*RECIP_BPF;
}
