module ArrayInterfaceStaticArraysCore

import StaticArraysCore, ArrayInterfaceCore, Adapt
using LinearAlgebra

function ArrayInterfaceCore.undefmatrix(::StaticArraysCore.MArray{S, T, N, L}) where {S, T, N, L}
    return StaticArraysCore.MMatrix{L, L, T, L*L}(undef)
end
# SArray doesn't have an undef constructor and is going to be small enough that this is fine.
function ArrayInterfaceCore.undefmatrix(s::StaticArraysCore.SArray)
    v = vec(s)
    return v.*v'
end

ArrayInterfaceCore.ismutable(::Type{<:StaticArraysCore.StaticArray}) = false
ArrayInterfaceCore.ismutable(::Type{<:StaticArraysCore.MArray}) = true
ArrayInterfaceCore.ismutable(::Type{<:StaticArraysCore.SizedArray}) = true

ArrayInterfaceCore.can_setindex(::Type{<:StaticArraysCore.StaticArray}) = false
ArrayInterfaceCore.buffer(A::Union{StaticArraysCore.SArray,StaticArraysCore.MArray}) = getfield(A, :data)

function ArrayInterfaceCore.lu_instance(_A::StaticArraysCore.StaticMatrix{N,N}) where {N}
    lu(one(_A))
end

function ArrayInterfaceCore.restructure(x::StaticArraysCore.SArray{S,T,N}, y::StaticArraysCore.SArray) where {S,T,N}
    StaticArraysCore.SArray{S,T,N}(y)
end
ArrayInterfaceCore.restructure(x::StaticArraysCore.SArray{S}, y) where {S} = StaticArraysCore.SArray{S}(y)

Adapt.adapt_storage(::Type{<:StaticArraysCore.SArray{S}}, xs::Array) where {S} = SArray{S}(xs)

end
