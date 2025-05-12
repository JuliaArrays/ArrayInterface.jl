module ArrayInterfaceSparseArraysExt

import ArrayInterface: buffer, has_sparsestruct, issingular, findstructralnz, bunchkaufman_instance, DEFAULT_CHOLESKY_PIVOT, cholesky_instance, ldlt_instance, lu_instance, qr_instance
using ArrayInterface.LinearAlgebra
using SparseArrays

buffer(x::SparseMatrixCSC) = getfield(x, :nzval)
buffer(x::SparseVector) = getfield(x, :nzval)
has_sparsestruct(::Type{<:SparseMatrixCSC}) = true
issingular(A::AbstractSparseMatrix) = !issuccess(lu(A, check = false))

function findstructralnz(x::SparseMatrixCSC)
    rowind, colind, _ = findnz(x)
    (rowind, colind)
end

function bunchkaufman_instance(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    bunchkaufman(SparseMatrixCSC{Tv, Ti}(similar(A, 1, 1)), check = false)
end

function cholesky_instance(A::Union{SparseMatrixCSC{Tv, Ti},Symmetric{<:Number,<:SparseMatrixCSC{Tv, Ti}}}, pivot = DEFAULT_CHOLESKY_PIVOT) where {Tv, Ti}
    cholesky(SparseMatrixCSC{Tv, Ti}(similar(A, 1, 1)), check = false)
end

function ldlt_instance(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    ldlt(SparseMatrixCSC{Tv, Ti}(similar(A, 1, 1)), check=false)
end

# Could be optimized but this should work for any real case.
function lu_instance(jac_prototype::SparseMatrixCSC{Tv, Ti}, pivot = DEFAULT_CHOLESKY_PIVOT) where {Tv, Ti}
    lu(SparseMatrixCSC{Tv, Ti}(rand(1,1)))
end

function qr_instance(jac_prototype::SparseMatrixCSC{Tv, Ti}, pivot = DEFAULT_CHOLESKY_PIVOT) where {Tv, Ti}
    qr(SparseMatrixCSC{Tv, Ti}(rand(1,1)))
end

end
