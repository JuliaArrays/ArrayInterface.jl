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

function bunchkaufman_instance(A::SparseMatrixCSC)
    bunchkaufman(sparse(similar(A, 1, 1)), check = false)
end

function cholesky_instance(A::Union{SparseMatrixCSC,Symmetric{<:Number,<:SparseMatrixCSC}}, pivot = DEFAULT_CHOLESKY_PIVOT)
    cholesky(sparse(similar(A, 1, 1)), check = false)
end

function ldlt_instance(A::SparseMatrixCSC)
    ldlt(sparse(similar(A, 1, 1)), check=false)
end

# Could be optimized but this should work for any real case.
function lu_instance(jac_prototype::SparseMatrixCSC, pivot = DEFAULT_CHOLESKY_PIVOT)
    lu(sparse(rand(1,1)))
end

function qr_instance(jac_prototype::SparseMatrixCSC, pivot = DEFAULT_CHOLESKY_PIVOT)
    qr(sparse(rand(1,1)))
end

end
