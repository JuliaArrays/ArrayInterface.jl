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
@propagate_inbounds function Base.getindex(x::ArrayIndex, i::CanonicalInt, ii::CanonicalInt...)
    x[NDIndex(i, ii...)]
end
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
    pdims::P

    AxisIndex(@nospecialize(i), @nospecialize(d)) = new{typeof(i),typeof(d),ndims_index(i)}(i, d)
    AxisIndex(@nospecialize(i::AxisIndex), @nospecialize(d)) = AxisIndex(getfield(i, :index), d)
    AxisIndex(@nospecialize(i)) = AxisIndex(i, nothing)
end

ndims_index(@nospecialize T::Type{<:AxisIndex}) = ndims_index(fieldtype(T, :index))
ndims_shape(@nospecialize T::Type{<:AxisIndex}) = ndims_shape(fieldtype(T, :index))
is_splat_index(@nospecialize T::Type{<:AxisIndex}) = is_splat_index(fieldtype(T, :index))
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
