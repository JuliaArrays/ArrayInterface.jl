
"""
    axes_types(::Type{T}, dim)

Returns the axis type along dimension `dim`.
"""
axes_types(x, dim) = axes_types(typeof(x), dim)
@inline axes_types(::Type{T}, dim) where {T} = axes_types(T, to_dims(T, dim))
@inline function axes_types(::Type{T}, dim::StaticInt{D}) where {T,D}
    if D > ndims(T)
        return OptionallyStaticUnitRange{One,One}
    else
        return _get_tuple(axes_types(T), dim)
    end
end
@inline function axes_types(::Type{T}, dim::Int) where {T}
    if dim > ndims(T)
        return OptionallyStaticUnitRange{One,One}
    else
        return axes_types(T).parameters[dim]
    end
end

"""
    axes_types(::Type{T}) -> Type

Returns the type of the axes for `T`
"""
axes_types(x) = axes_types(typeof(x))
axes_types(::Type{T}) where {T<:Array} = Tuple{Vararg{OneTo{Int},ndims(T)}}
function axes_types(::Type{T}) where {T}
    if parent_type(T) <: T
        return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},ndims(T)}}
    else
        return axes_types(parent_type(T))
    end
end
axes_types(::Type{LinearIndices{N,R}}) where {N,R} = R
axes_types(::Type{CartesianIndices{N,R}}) where {N,R} = R
function axes_types(::Type{T}) where {T<:Union{Adjoint,Transpose}}
    P = parent_type(T)
    return Tuple{axes_types(P, static(2)), axes_types(P, static(1))}
end
function axes_types(::Type{T}) where {T<:PermutedDimsArray}
    return eachop_tuple(_get_tuple, to_parent_dims(T), axes_types(parent_type(T)))
end
function axes_types(::Type{T}) where {T<:AbstractRange}
    if known_length(T) === nothing
        return Tuple{OptionallyStaticUnitRange{One,Int}}
    else
        return Tuple{OptionallyStaticUnitRange{One,StaticInt{known_length(T)}}}
    end
end
function axes_types(::Type{T}) where {N,T<:Base.ReshapedArray{<:Any,N}}
    return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},N}}
end

@inline function axes_types(::Type{T}) where {N,P,I,T<:SubArray{<:Any,N,P,I}}
    return eachop_tuple(_sub_axis_type, to_parent_dims(T), T)
end
@inline function _sub_axis_type(::Type{A}, dim::StaticInt) where {T,N,P,I,A<:SubArray{T,N,P,I}}
    return OptionallyStaticUnitRange{
        _int_or_static_int(known_first(axes_types(P, dim))),
        _int_or_static_int(known_length(_get_tuple(I, dim)))
    }
end

function axes_types(::Type{R}) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    if _is_reshaped(R)
        if sizeof(S) === sizeof(T)
            return axes_types(A)
        elseif sizeof(S) > sizeof(T)
            return eachop_tuple(_reshaped_axis_type, to_parent_dims(R), R)
        else
            return eachop_tuple(axes_types, to_parent_dims(R), A)
        end
    else
        return eachop_tuple(_non_reshaped_axis_type, to_parent_dims(R), R)
    end
end

function _reshaped_axis_type(::Type{R}, dim::StaticInt) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    return axes_types(parent_type(R), dim)
end
function _reshaped_axis_type(::Type{R}, dim::Zero) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    return OptionallyStaticUnitRange{One,StaticInt{div(sizeof(S), sizeof(T))}}
end

function _non_reshaped_axis_type(::Type{R}, dim::StaticInt) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    return axes_types(parent_type(R), dim)
end
function _non_reshaped_axis_type(::Type{R}, dim::One) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    paxis = axes_types(A, dim)
    len = known_length(paxis)
    if len === nothing
        raxis = OptionallyStaticUnitRange{One,Int}
    else
        raxis = OptionallyStaticUnitRange{One,StaticInt{div(len * sizeof(S), sizeof(T))}}
    end
    return similar_type(paxis, Int, raxis)
end

#=
    similar_type(orignal_type, new_data_type)
=#
similar_type(::Type{OneTo{Int}}, ::Type{Int}, ::Type{OneTo{Int}}) = OneTo{Int}
similar_type(::Type{OneTo{Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,Int}}) = OneTo{Int}
similar_type(::Type{OneTo{Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}) where {N} = OptionallyStaticUnitRange{One,StaticInt{N}}

similar_type(::Type{OptionallyStaticUnitRange{One,Int}}, ::Type{Int}, ::Type{OneTo{Int}}) = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,Int}}) = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}) where {N} = OptionallyStaticUnitRange{One,StaticInt{N}}

similar_type(::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}, ::Type{Int}, ::Type{OneTo{Int}}) where {N} = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,Int}}) where {N} = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,StaticInt{N1}}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,StaticInt{N2}}}) where {N1,N2} = OptionallyStaticUnitRange{One,StaticInt{N2}}

"""
    axes(A, d)

Return a valid range that maps to each index along dimension `d` of `A`.
"""
@inline axes(a, dim) = axes(a, to_dims(a, dim))
@inline axes(a, dims::Tuple{Vararg{Any,K}}) where {K} = (axes(a, first(dims)), axes(a, tail(dims))...)
@inline axes(a, dims::Tuple{T}) where {T} = (axes(a, first(dims)), )
@inline axes(a, ::Tuple{}) = ()
@inline function _axes(a::A, dim::Integer) where {A}
    if parent_type(A) <: A
        return Base.axes(a, Int(dim))
    else
        return _axes(parent(a), to_parent_dims(A, dim))
    end
end
@inline function _axes(A::CartesianIndices{N}, dim::Integer) where {N}
    if dim > N
        return static(1):static(1)
    else
        return getfield(axes(A), Int(dim))
    end
end
@inline function _axes(A::LinearIndices{N}, dim::Integer) where {N}
    if dim > N
        return static(1):static(1)
    else
        return getfield(axes(A), Int(dim))
    end
end
@inline _axes(::LinearAlgebra.AdjOrTrans{T,V}, ::One) where {T,V<:AbstractVector} = One():One()
@inline axes(A::AbstractArray, dim::Integer) = _axes(A, dim, False())
@inline axes(A::AbstractArray{T,N}, ::StaticInt{M}) where {T,N,M} = _axes(A, StaticInt{M}(), gt(StaticInt{M}(),StaticInt{N}()))
@inline _axes(::Any, ::Any, ::True) = One():One()
@inline _axes(A::AbstractArray, dim, ::False) = _axes(A, dim)


@inline _axes(A::SubArray, dim::Integer) = Base.axes(A, Int(dim))  # TODO implement ArrayInterface version
@inline _axes(A::ReinterpretArray, dim::Integer) = Base.axes(A, Int(dim))  # TODO implement ArrayInterface version
@inline _axes(A::Base.ReshapedArray, dim::Integer) = Base.axes(A, Int(dim))  # TODO implement ArrayInterface version

"""
    axes(A)

Return a tuple of ranges where each range maps to each element along a dimension of `A`.
"""
@inline function axes(a::A) where {A}
    if parent_type(A) <: A
        return Base.axes(a)
    else
        return axes(parent(a))
    end
end
axes(A::PermutedDimsArray) = permute(axes(parent(A)), to_parent_dims(A))
axes(A::Union{Transpose,Adjoint}) = _axes(A, parent(A))
_axes(A::Union{Transpose,Adjoint}, p::AbstractVector) = (One():One(), axes(p, One()))
_axes(A::Union{Transpose,Adjoint}, p::AbstractMatrix) = (axes(p, StaticInt(2)), axes(p, One()))
axes(A::SubArray) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::ReinterpretArray) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::Base.ReshapedArray) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::CartesianIndices) = A.indices
axes(A::LinearIndices) = A.indices

"""
    LazyAxis{N}(parent::AbstractArray)

A lazy representation of `axes(parent, N)`.
"""
struct LazyAxis{N,P} <: AbstractUnitRange{Int}
    parent::P

    LazyAxis{N}(parent::P) where {N,P} = new{N::Int,P}(parent)
    @inline function LazyAxis{:}(parent::P) where {P}
        if ndims(P) === 1
            return new{1,P}(parent)
        else
            return new{:,P}(parent)
        end
    end
end

@inline Base.parent(x::LazyAxis{N,P}) where {N,P} = axes(getfield(x, :parent), static(N))
@inline function Base.parent(x::LazyAxis{:,P}) where {P}
    return eachindex(IndexLinear(), getfield(x, :parent))
end

@inline parent_type(::Type{LazyAxis{N,P}}) where {N,P} = axes_types(P, static(N))
# TODO this approach to parent_type(::Type{LazyAxis{:}}) is a bit hacky. Something like
# LabelledArrays has a linear set of symbolic keys, which could be propagated through
# `to_indices` for key based indexing. However, there currently isn't a good way of handling
# that when the linear indices aren't linearly accessible through a child array (e.g, adjoint)
# For now we just make sure the linear elements are accurate.
parent_type(::Type{LazyAxis{:,P}}) where {P<:Array} = OneTo{Int}
@inline function parent_type(::Type{LazyAxis{:,P}}) where {P}
    if known_length(P) === nothing
        return OptionallyStaticUnitRange{StaticInt{1},Int}
    else
        return OptionallyStaticUnitRange{StaticInt{1},StaticInt{known_length(P)}}
    end
end

Base.keys(x::LazyAxis) = keys(parent(x))

Base.IndexStyle(::Type{T}) where {T<:LazyAxis} = IndexStyle(parent_type(T))

can_change_size(::Type{LazyAxis{N,P}}) where {N,P} = can_change_size(P)

known_first(::Type{T}) where {T<:LazyAxis} = known_first(parent_type(T))

known_length(::Type{LazyAxis{N,P}}) where {N,P} = known_size(P, N)
known_length(::Type{LazyAxis{:,P}}) where {P} = known_length(P)

@inline function known_last(::Type{T}) where {T<:LazyAxis}
    return _lazy_axis_known_last(known_first(T), known_length(T))
end
_lazy_axis_known_last(start::Int, length::Int) = (length + start) - 1
_lazy_axis_known_last(::Any, ::Any) = nothing

@inline function Base.first(x::LazyAxis{N})::Int where {N}
    if known_first(x) === nothing
        return offsets(getfield(x, :parent), static(N))
    else
        return known_first(x)
    end
end
@inline function Base.first(x::LazyAxis{:})::Int
    if known_first(x) === nothing
        return firstindex(getfield(x, :parent))
    else
        return known_first(x)
    end
end

@inline function Base.length(x::LazyAxis{N})::Int where {N}
    if known_length(x) === nothing
        return size(getfield(x, :parent), static(N))
    else
        return known_length(x)
    end
end
@inline function Base.length(x::LazyAxis{:})::Int
    if known_length(x) === nothing
        return lastindex(getfield(x, :parent))
    else
        return known_length(x)
    end
end

@inline function Base.last(x::LazyAxis)::Int
    if known_last(x) === nothing
        if known_first(x) === 1
            return length(x)
        else
            return (static_length(x) + static_first(x)) - 1
        end
    else
        return known_last(x)
    end
end

Base.to_shape(x::LazyAxis) = length(x)

@inline function Base.checkindex(::Type{Bool}, x::LazyAxis, i::Integer)
    if known_first(x) === nothing || known_last(x) === nothing
        return checkindex(Bool, parent(x), i)
    else  # everything is static so we don't have to retrieve the axis
        return (!(known_first(x) > i) || !(known_last(x) < i))
    end
end

@propagate_inbounds function Base.getindex(x::LazyAxis, i::Integer)
    @boundscheck checkindex(Bool, x, i) || throw(BoundsError(x, i))
    return Int(i)
end
@propagate_inbounds Base.getindex(x::LazyAxis, i::StepRange{T}) where {T<:Integer} = parent(x)[i]
@propagate_inbounds Base.getindex(x::LazyAxis, i::AbstractUnitRange{<:Integer}) = parent(x)[i]

Base.show(io::IO, x::LazyAxis{N}) where {N} = print(io, "LazyAxis{$N}($(parent(x))))")

"""
    lazy_axes(x)

Produces a tuple of axes where each axis is constructed lazily. If an axis of `x` is already
constructed or it is simply retrieved.
"""
@generated function lazy_axes(x::X) where {X}
    Expr(:block,
         Expr(:meta, :inline),
         Expr(:tuple, [:(LazyAxis{$dim}(x)) for dim in 1:ndims(X)]...)
    )
end
lazy_axes(x::LinearIndices) = axes(x)
lazy_axes(x::CartesianIndices) = axes(x)
@inline lazy_axes(x::MatAdjTrans) = reverse(lazy_axes(parent(x)))
@inline lazy_axes(x::VecAdjTrans) = (LazyAxis{1}(x), first(lazy_axes(parent(x))))
@inline lazy_axes(x::PermutedDimsArray) = permute(lazy_axes(parent(x)), to_parent_dims(x))

