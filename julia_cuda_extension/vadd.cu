extern "C"
{
  __global__ void vadd(const float *a, const float *b, float *c, int n)
  {
    int i= threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
    { 
      c[i]=a[i]+b[i];
    }
  }
}

