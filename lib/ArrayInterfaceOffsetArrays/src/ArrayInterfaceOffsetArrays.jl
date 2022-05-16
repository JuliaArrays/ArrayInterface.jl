module ArrayInterfaceOffsetArrays

using ArrayInterfaceCore
using OffsetArrays
using Static

relative_offsets(r::OffsetArrays.IdOffsetRange) = (getfield(r, :offset),)
relative_offsets(A::OffsetArrays.OffsetArray) = getfield(A, :offsets)
function relative_offsets(A::OffsetArrays.OffsetArray, ::StaticInt{dim}) where {dim}
    if dim > ndims(A)
        return static(0)
    else
        return getfield(relative_offsets(A), dim)
    end
end
function relative_offsets(A::OffsetArrays.OffsetArray, dim::Int)
    if dim > ndims(A)
        return 0
    else
        return getfield(relative_offsets(A), dim)
    end
end
ArrayInterfaceCore.parent_type(::Type{<:OffsetArrays.OffsetArray{T,N,A}}) where {T,N,A} = A
function _offset_axis_type(::Type{T}, dim::StaticInt{D}) where {T,D}
    OffsetArrays.IdOffsetRange{Int,ArrayInterfaceCore.axes_types(T, dim)}
end
function ArrayInterfaceCore.axes_types(::Type{T}) where {T<:OffsetArrays.OffsetArray}
    Static.eachop_tuple(_offset_axis_type, Static.nstatic(Val(ndims(T))), ArrayInterfaceCore.parent_type(T))
end
function ArrayInterfaceCore.known_offsets(::Type{A}) where {A<:OffsetArrays.OffsetArray}
    ntuple(identity -> nothing, Val(ndims(A)))
end
function ArrayInterfaceCore.offsets(A::OffsetArrays.OffsetArray)
    map(+, ArrayInterfaceCore.offsets(parent(A)), relative_offsets(A))
end
@inline function ArrayInterfaceCore.offsets(A::OffsetArrays.OffsetArray, dim)
    d = ArrayInterfaceCore.to_dims(A, dim)
    ArrayInterfaceCore.offsets(parent(A), d) + relative_offsets(A, d)
end
@inline function ArrayInterfaceCore.axes(A::OffsetArrays.OffsetArray)
    map(OffsetArrays.IdOffsetRange, ArrayInterfaceCore.axes(parent(A)), relative_offsets(A))
end
@inline function ArrayInterfaceCore.axes(A::OffsetArrays.OffsetArray, dim)
    d = ArrayInterfaceCore.to_dims(A, dim)
    OffsetArrays.IdOffsetRange(ArrayInterfaceCore.axes(parent(A), d), relative_offsets(A, d))
end

end # module
