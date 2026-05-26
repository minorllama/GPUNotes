
using CUDA

libcppexts = "./libcppexts.so"

A = CuArray{Float32}([1, 2, 3, 4, 1, 2, 3, 4])
B = CuArray{Float32}([2, 2, 2, 2, 4, 4, 4, 4])
C = CUDA.zeros(Float32, 2)
dim = 4
N = 2
warpsize = 32
threadsPerBlock = warpsize
numBlocks = 2

starts = CuArray{Int32}(Array([1, 5]))
stops = CuArray{Int32}(Array([4, 8]))

println(C)
ccall((:launch_batched_dot_product, libcppexts), Cvoid,
      (CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32}, Cint, Cint, CuPtr{Int32}, CuPtr{Int32}, Cint, Cint),
      C, A, B, N, dim, starts, stops, threadsPerBlock, numBlocks)
println(C) 

for i in 1:N 
    cpu = CUDA.@allowscalar sum(A[starts[i]:stops[i]].* B[starts[i]:stops[i]])
    CUDA.@allowscalar println(i, ":",  cpu, " ", C[i])
end

#=

Float32[0.0, 0.0]
(cfg:32,2,32)
2,4
Float32[20.0, 40.0]
1:20.0 20.0
2:40.0 40.0
ok

=#



