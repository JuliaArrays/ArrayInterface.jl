module ArrayInterfaceStaticArraysCoreExt

if isdefined(Base, :get_extension)
    import ArrayInterface
    using LinearAlgebra
    import StaticArraysCore
else
    import ..ArrayInterface
    using ..LinearAlgebra
    import ..StaticArraysCore
end

function ArrayInterface.undefmatrix(::StaticArraysCore.MArray{S, T, N, L}) where {S, T, N, L}
    return StaticArraysCore.MMatrix{L, L, T, L*L}(undef)
end
# SArray doesn't have an undef constructor and is going to be small enough that this is fine.
function ArrayInterface.undefmatrix(s::StaticArraysCore.SArray)
    v = vec(s)
    return v.*v'
end

ArrayInterface.ismutable(::Type{<:StaticArraysCore.StaticArray}) = false
ArrayInterface.ismutable(::Type{<:StaticArraysCore.MArray}) = true
ArrayInterface.ismutable(::Type{<:StaticArraysCore.SizedArray}) = true

ArrayInterface.can_setindex(::Type{<:StaticArraysCore.StaticArray}) = false
ArrayInterface.buffer(A::Union{StaticArraysCore.SArray,StaticArraysCore.MArray}) = getfield(A, :data)

function ArrayInterface.lu_instance(_A::StaticArraysCore.StaticMatrix{N,N}) where {N}
    lu(one(_A))
end

ArrayInterface.restructure(x::StaticArraysCore.SArray{S}, y) where {S} = StaticArraysCore.SArray{S}(y)

end
