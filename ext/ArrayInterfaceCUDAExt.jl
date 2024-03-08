module ArrayInterfaceCUDAExt

using ArrayInterface
using CUDA
using CUDA.CUSOLVER
using LinearAlgebra

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    if VERSION >= v"1.8-"
        ipiv = cu(Vector{Int32}(undef, 0))
        info = zero(Int)
        return LinearAlgebra.LU(similar(A, 0, 0), ipiv, info)
    else
        LinearAlgebra.lu(A; check = false)
    end
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

end # module
