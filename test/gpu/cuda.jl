using CUDSS, CUDA, SparseArrays, LinearAlgebra
using CUDA.CUSPARSE
using ArrayInterface

using Test

A_cpu = Float32[1 0; 0 1]
A_dense = CuMatrix(A_cpu)
A_sparse = CuSparseMatrixCSR(sparse(A_cpu))

# Test whether lu_instance throws an error when invoked with an gpu array
lu_inst_dense = ArrayInterface.lu_instance(A_dense)
lu_inst_sparse = ArrayInterface.lu_instance(A_sparse)

# test that lu! is valid when using the inst as scratch
lu_sparse = lu!(lu_inst_sparse, A_sparse)

#test that the resulting lu works
b = CuVector([1f0, 1f0])
@test CUDA.@allowscalar lu_sparse \ b == [1, 1]
