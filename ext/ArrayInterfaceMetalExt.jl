module ArrayInterfaceMetalExt

using ArrayInterface
using Metal
using LinearAlgebra

function ArrayInterface.lu_instance(A::MtlMatrix{T}) where {T}
    ipiv = MtlVector{Int32}(undef, 0)
    info = zero(Int)
    return LinearAlgebra.LU(similar(A, 0, 0), ipiv, info)
end

ArrayInterface.device(::Type{<:Metal.MtlArray}) = ArrayInterface.GPU()

end # module