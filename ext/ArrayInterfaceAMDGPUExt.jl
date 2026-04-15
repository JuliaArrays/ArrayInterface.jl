module ArrayInterfaceAMDGPUExt

using ArrayInterface
using AMDGPU
using LinearAlgebra

function ArrayInterface.lu_instance(A::ROCMatrix{T}) where {T}
    ipiv = ROCVector{Cint}(undef, 0)
    info = zero(Int)
    return LinearAlgebra.LU(similar(A, 0, 0), ipiv, info)
end

ArrayInterface.device(::Type{<:AMDGPU.ROCArray}) = ArrayInterface.GPU()

function ArrayInterface.promote_eltype(
        ::Type{<:AMDGPU.ROCArray{T, N, B}}, ::Type{T2}
    ) where {T, N, B, T2}
    return AMDGPU.ROCArray{promote_type(T, T2), N, B}
end

end # module
