module ArrayInterfaceStaticArraysCoreExt

import ArrayInterface
using LinearAlgebra
import StaticArraysCore: SArray, SMatrix, SVector, StaticMatrix, StaticArray, SizedArray, MArray, MMatrix

function ArrayInterface.undefmatrix(::MArray{S, T, N, L}) where {S, T, N, L}
    return MMatrix{L, L, T, L*L}(undef)
end
# SArray doesn't have an undef constructor and is going to be small enough that this is fine.
function ArrayInterface.undefmatrix(s::SArray)
    v = vec(s)
    return v.*v'
end

ArrayInterface.ismutable(::Type{<:StaticArray}) = false
ArrayInterface.ismutable(::Type{<:MArray}) = true
ArrayInterface.ismutable(::Type{<:SizedArray}) = true

ArrayInterface.can_setindex(::Type{<:StaticArray}) = false
ArrayInterface.can_setindex(::Type{<:MArray}) = true
ArrayInterface.buffer(A::Union{SArray, MArray}) = getfield(A, :data)

function ArrayInterface.lu_instance(A::StaticMatrix{N,N}) where {N}
    lu(one(A))
end

ArrayInterface.restructure(x::SArray{S}, y) where {S} = SArray{S}(y)

end
