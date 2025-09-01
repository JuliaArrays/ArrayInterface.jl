using Metal
using ArrayInterface

using Test

# Test whether lu_instance throws an error when invoked with a Metal.jl gpu array
@test !isa(try ArrayInterface.lu_instance(MtlArray([1.f0 1.f0; 1.f0 1.f0])) catch ex ex end, Exception) 