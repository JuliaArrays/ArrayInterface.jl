module FixedSizeArraysExt

using ArrayInterface: ArrayInterface
using FixedSizeArrays: FixedSizeMatrix
using LinearAlgebra: lu, qr

# Factorizing an empty matrix is cheap and yields exactly the factorization type
# of `A` (pivot vector types differ across eltypes, e.g. for BigFloat).
ArrayInterface.lu_instance(A::FixedSizeMatrix) = lu(similar(A, 0, 0), check = false)

function ArrayInterface.qr_instance(A::FixedSizeMatrix, pivot = ArrayInterface.DEFAULT_CHOLESKY_PIVOT)
    return qr(similar(A, 0, 0), pivot)
end

end
