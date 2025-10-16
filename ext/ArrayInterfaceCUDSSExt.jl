module ArrayInterfaceCUDSSExt

using ArrayInterface
using CUDSS
using CUDA

function ArrayInterface.lu_instance(A::CUDSS.CuSparseMatrixCSR)
    ArrayInterface.LinearAlgebra.checksquare(A)
    T = eltype(A)
    n = size(A, 1)

    # Use standard CUDA types (CuVector) instead of deprecated CudssMatrix
    x = CUDA.CuVector{T}(undef, n)
    b = CUDA.CuVector{T}(undef, n)

    fact = CudssSolver(A, "G", 'F')
    cudss("analysis", fact, x, b)
    fact
end

end
