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

end # module
