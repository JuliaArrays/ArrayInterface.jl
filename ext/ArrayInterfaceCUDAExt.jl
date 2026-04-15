module ArrayInterfaceCUDAExt

using ArrayInterface
using CUDA
using CUDA.CUSOLVER
using LinearAlgebra

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    ipiv = cu(Vector{Int32}(undef, 0))
    info = zero(Int)
    return LinearAlgebra.LU(similar(A, 0, 0), ipiv, info)
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

function ArrayInterface.promote_eltype(
        ::Type{<:CUDA.CuArray{T, N, M}}, ::Type{T2}
    ) where {T, N, M, T2}
    return CUDA.CuArray{promote_type(T, T2), N, M}
end

end # module
