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

function ArrayInterface.promote_eltype(
        ::Type{<:Metal.MtlArray{T, N, S}}, ::Type{T2}
    ) where {T, N, S, T2}
    return Metal.MtlArray{promote_type(T, T2), N, S}
end

end # module