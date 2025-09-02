using Metal
using ArrayInterface
using LinearAlgebra

using Test

# Test that lu_instance works with Metal.jl gpu arrays
@test isa(ArrayInterface.lu_instance(MtlArray([1.f0 1.f0; 1.f0 1.f0])), LU) 