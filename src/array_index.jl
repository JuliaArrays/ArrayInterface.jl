"""
    StrideIndex(x)

Subtype of `ArrayIndex` that transforms and index using stride layout information
derived from `x`.
"""
struct StrideIndex{N,R,C,S,O} <: ArrayIndex{N}
    strides::S
    offsets::O

    @inline function StrideIndex{N,R,C}(s::S, o::O) where {N,R,C,S,O}
        return new{N,R::NTuple{N,Int},C,S,O}(s, o)
    end
    @inline function StrideIndex{N,R,C}(a::A) where {N,R,C,A}
        return StrideIndex{N,R,C}(strides(a), offsets(a))
    end
    @inline function StrideIndex(a::A) where {A}
        return StrideIndex{ndims(A),known(stride_rank(A)),known(contiguous_axis(A))}(a)
    end
end

## getindex
@propagate_inbounds Base.getindex(x::ArrayIndex, i::CanonicalInt, ii::CanonicalInt...) = x[NDIndex(i, ii...)]

@inline function Base.getindex(x::StrideIndex{N}, i::AbstractCartesianIndex) where {N}
    return _strides2int(offsets(x), strides(x), Tuple(i)) + static(1)
end
@generated function _strides2int(o::O, s::S, i::I) where {O,S,I}
    N = known_length(S)
    out = :()
    for i in 1:N
        tmp = :(((getfield(i, $i) - getfield(o, $i)) * getfield(s, $i)))
        out = ifelse(i === 1, tmp, :($out + $tmp))
    end
    return Expr(:block, Expr(:meta, :inline), out)
end

struct AxisIndex{I,P,N} <: ArrayIndex{N}
    index::I
    parentdims::P

    AxisIndex(@nospecialize(i), @nospecialize(d)) = new{typeof(i),typeof(d),ndims_index(i)}(i, d)
    AxisIndex(@nospecialize(i::AxisIndex), @nospecialize(d)) = AxisIndex(getfield(i, :index), d)
    AxisIndex(@nospecialize(i)) = AxisIndex(i, nothing)
end

parentdims(@nospecialize i::AxisIndex) = getfield(i, :parentdims)

struct Begin <: ArrayIndex{1} end

struct End <: ArrayIndex{1} end

ndims_index(@nospecialize T::Type{<:AxisIndex}) = ndims_index(fieldtype(T, :index))
ndims_index(::Type{Begin}) = 1
ndims_index(::Type{End}) = 1

ndims_shape(@nospecialize T::Type{<:AxisIndex}) = ndims_shape(fieldtype(T, :index))
ndims_shape(::Type{Begin}) = 0
ndims_shape(::Type{End}) = 0

Base.:(:)(::Begin, stop) = AxisIndex(<=(stop))
Base.:(:)(start, ::End) = AxisIndex(>=(start))
Base.:(:)(::Begin, ::End) = (:)

is_splat_index(@nospecialize T::Type{<:AxisIndex}) = is_splat_index(fieldtype(T, :index))

to_indices(A, ::Tuple{}) = ()
@inline to_indices(A, inds::Tuple{Vararg{Any}}) = Base.to_indices(A, as_indices(A, inds))

Base.to_index(A, @nospecialize(i::AxisIndex{<:Union{CartesianIndices{0,Tuple{}},Base.Slice,StaticInt,AbstractArray{<:Integer},AbstractArray{<:AbstractCartesianIndex}}})) = getfield(i, :index)
# FIXME better tracking of trailing dimensions
@inline function as_indices(::A, inds::I) where {A,I}
    minfo = map_indices_info(IndicesInfo{ndims(A)}(I))
    ntuple(Val{nfields(minfo)}()) do i
        pdim = getfield(getfield(minfo, i), 2)
        if pdim === StaticInt(0)
            AxisIndex(getfield(inds, i), StaticInt(ndims(A) + 1))
        else
            AxisIndex(getfield(inds, i), pdim)
        end
    end
end
@inline function Base.to_indices(A, inds::Tuple{Vararg{ArrayIndex}})
    Base.to_indices(A, as_indices(A, inds))
end
@inline function Base.to_indices(A, inds::Tuple{Vararg{AxisIndex{<:Any,<:Union{StaticInt,Tuple,Colon}}}})
    flatten_tuples(map(Fix1(Base.to_index, A), inds))
end

@inline Base.to_index(A, i::AxisIndex{Colon}) = indices(lazy_axes(A, parentdims(i)))
@inline function Base.to_index(A, @nospecialize(i::AxisIndex{<:Union{CartesianIndex,NDIndex,CartesianIndices}}))
    getfield(getfield(i, :index), 1)
end
@inline Base.to_index(A, i::AxisIndex{<:Base.BitInteger}) = Int(getfield(i, :index))
@inline function Base.to_index(A, i::AxisIndex{<:AbstractArray{Bool}})
    if (last(parentdims(i)) == ndims(A)) && (IndexStyle(A) isa IndexLinear)
        return LogicalIndex{Int}(getfield(i, :index))
    else
        return LogicalIndex(getfield(i, :index))
    end
end
@inline Base.to_index(A, i::AxisIndex{Begin}) = static_first(lazy_axes(A, parentdims(i)))
@inline Base.to_index(A, i::AxisIndex{End}) = static_last(lazy_axes(A, parentdims(i)))
@inline function Base.to_index(A, i::AxisIndex{<:Fix2{<:Union{typeof(<),typeof(isless)},<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, parentdims(i))
    static_first(x):min(_sub1(canonicalize(getfield(i, :index).x)), static_last(x))
end
@inline function Base.to_index(A, i::AxisIndex{<:Fix2{typeof(<=),<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, parentdims(i))
    static_first(x):min(canonicalize(getfield(i, :index).x), static_last(x))
end
@inline function Base.to_index(A, i::AxisIndex{<:Fix2{typeof(>=),<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, parentdims(i))
    max(canonicalize(getfield(i, :index).x), static_first(x)):static_last(x)
end
@inline function Base.to_index(A, i::AxisIndex{<:Fix2{typeof(>),<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, getfield(i, :parentdims))
    max(_add1(canonicalize(getfield(i, :index).x)), static_first(x)):static_last(x)
end
@inline function Base.to_index(A, i::AxisIndex)
    to_index(CartesianIndices(lazy_axes(A, parentdims(i))), getfield(i, :index))
end
