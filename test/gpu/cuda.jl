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

# CUSPARSE arrays have no `setindex!`, so `can_setindex` must not claim they do
@test ArrayInterface.can_setindex(A_dense)
@test !ArrayInterface.can_setindex(A_sparse)
@test !ArrayInterface.can_setindex(CuSparseMatrixCSC(sparse(A_cpu)))
@test !ArrayInterface.can_setindex(CuSparseMatrixCOO(sparse(A_cpu)))
@test !ArrayInterface.can_setindex(CuSparseVector(sparsevec([1], [1.0f0], 2)))
@test_throws CanonicalIndexError A_sparse[1, 2] = 1.0f0
