using CUDA
using ArrayInterface

using Test

# Test whether lu_instance throws an error when invoked with an gpu array
@test !isa(try ArrayInterface.lu_instance(CUDA.CuArray([1.f0 1.f0; 1.f0 1.f0])) catch ex ex end, Exception) 
