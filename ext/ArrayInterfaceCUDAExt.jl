module ArrayInterfaceCUDAExt

using ArrayInterface
using CUDA
using CUDA.CUSOLVER
using CUDA.CUSPARSE
using LinearAlgebra

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    ipiv = cu(Vector{Int32}(undef, 0))
    info = zero(Int)
    return LinearAlgebra.LU(similar(A, 0, 0), ipiv, info)
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

# CUSPARSE arrays implement no `setindex!` at all, so the `true` default is wrong
# for them and callers guarding mutation with `can_setindex` hit a CanonicalIndexError.
const CuSparseArray = Union{
    CUSPARSE.CuSparseVector,
    CUSPARSE.CuSparseMatrixCSC,
    CUSPARSE.CuSparseMatrixCSR,
    CUSPARSE.CuSparseMatrixBSR,
    CUSPARSE.CuSparseMatrixCOO,
}
ArrayInterface.can_setindex(::Type{<:CuSparseArray}) = false

# GPU CSC/CSR do not subtype `SparseArrays.AbstractSparseMatrixCSC`, so they need their own
# structure comparison. The stored index arrays are `CuVector`s; `==` reduces on-device.
function ArrayInterface.same_sparsity_structure(
        A::CUSPARSE.CuSparseMatrixCSC, B::CUSPARSE.CuSparseMatrixCSC
    )
    size(A) == size(B) && A.colPtr == B.colPtr && A.rowVal == B.rowVal
end
function ArrayInterface.same_sparsity_structure(
        A::CUSPARSE.CuSparseMatrixCSR, B::CUSPARSE.CuSparseMatrixCSR
    )
    size(A) == size(B) && A.rowPtr == B.rowPtr && A.colVal == B.colVal
end

function ArrayInterface.promote_eltype(
        ::Type{<:CUDA.CuArray{T, N, M}}, ::Type{T2}
    ) where {T, N, M, T2}
    return CUDA.CuArray{promote_type(T, T2), N, M}
end

end # module
