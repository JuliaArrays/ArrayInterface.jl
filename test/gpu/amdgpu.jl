using AMDGPU
using ArrayInterface
using LinearAlgebra

using Test

A = ROCMatrix(Float32[1 0; 0 1])

# Test that lu_instance works with AMDGPU.jl ROC arrays
@test isa(ArrayInterface.lu_instance(A), LU)

# Test that device returns GPU()
@test ArrayInterface.device(typeof(A)) == ArrayInterface.GPU()
