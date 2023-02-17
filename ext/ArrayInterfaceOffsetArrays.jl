module ArrayInterfaceOffsetArraysExt

using ArrayInterface
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
ArrayInterface.parent_type(::Type{<:OffsetArrays.OffsetArray{T,N,A}}) where {T,N,A} = A
function _offset_axis_type(::Type{T}, dim::StaticInt{D}) where {T,D}
    OffsetArrays.IdOffsetRange{Int,ArrayInterface.axes_types(T, dim)}
end
function ArrayInterface.axes_types(::Type{T}) where {T<:OffsetArrays.OffsetArray}
    Static.eachop_tuple(
        _offset_axis_type,
        ntuple(static, StaticInt(ndims(T))),
        ArrayInterface.parent_type(T)
    )
end
ArrayInterface.strides(A::OffsetArray) = ArrayInterface.strides(parent(A))
function ArrayInterface.known_offsets(::Type{A}) where {A<:OffsetArrays.OffsetArray}
    ntuple(identity -> nothing, Val(ndims(A)))
end
function ArrayInterface.offsets(A::OffsetArrays.OffsetArray)
    map(+, ArrayInterface.offsets(parent(A)), relative_offsets(A))
end
@inline function ArrayInterface.offsets(A::OffsetArrays.OffsetArray, dim)
    d = ArrayInterface.to_dims(A, dim)
    ArrayInterface.offsets(parent(A), d) + relative_offsets(A, d)
end
@inline function ArrayInterface.axes(A::OffsetArrays.OffsetArray)
    map(OffsetArrays.IdOffsetRange, ArrayInterface.axes(parent(A)), relative_offsets(A))
end
@inline function ArrayInterface.axes(A::OffsetArrays.OffsetArray, dim)
    d = ArrayInterface.to_dims(A, dim)
    OffsetArrays.IdOffsetRange(ArrayInterface.axes(parent(A), d), relative_offsets(A, d))
end
function ArrayInterface.stride_rank(T::Type{<:OffsetArray})
  ArrayInterface.stride_rank(ArrayInterface.parent_type(T))
end
function ArrayInterface.dense_dims(T::Type{<:OffsetArray})
    ArrayInterface.dense_dims(ArrayInterface.parent_type(T))
end
function ArrayInterface.contiguous_axis(T::Type{<:OffsetArray})
  ArrayInterface.contiguous_axis(ArrayInterface.parent_type(T))
end

end # module
