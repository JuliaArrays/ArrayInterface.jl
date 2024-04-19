module ArrayInterfaceCUDSSExt

using ArrayInterface
using CUDSS

function ArrayInterface.lu_instance(A::CUDSS.CuSparseMatrixCSR)
    ArrayInterface.LinearAlgebra.checksquare(A)
    fact = CudssSolver(A, "G", 'F')
    T = eltype(A)
    n = size(A,1)
    x = CudssMatrix(T, n)
    b = CudssMatrix(T, n)
    cudss("analysis", fact, x, b)
    fact
end

end
