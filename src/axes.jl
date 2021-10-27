
"""
    axes_types(::Type{T}) -> Type{Tuple{Vararg{AbstractUnitRange{Int}}}}
    axes_types(::Type{T}, dim) -> Type{AbstractUnitRange{Int}}

Returns the type of each axis for the `T`, or the type of of the axis along dimension `dim`.
"""
axes_types(x, dim) = axes_types(typeof(x), dim)
@inline axes_types(::Type{T}, dim) where {T} = axes_types(T, to_dims(T, dim))
@inline function axes_types(::Type{T}, dim::StaticInt{D}) where {T,D}
    if D > ndims(T)
        return SOneTo{1}
    else
        return _get_tuple(axes_types(T), dim)
    end
end
@inline function axes_types(::Type{T}, dim::Int) where {T}
    if dim > ndims(T)
        return SOneTo{1}
    else
        return axes_types(T).parameters[dim]
    end
end
axes_types(x) = axes_types(typeof(x))
axes_types(::Type{T}) where {T<:Array} = NTuple{ndims(T),OneTo{Int}}
function axes_types(::Type{T}) where {T}
    if parent_type(T) <: T
        return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},ndims(T)}}
    else
        return axes_types(parent_type(T))
    end
end
axes_types(::Type{LinearIndices{N,R}}) where {N,R} = R
axes_types(::Type{CartesianIndices{N,R}}) where {N,R} = R
function axes_types(::Type{T}) where {T<:VecAdjTrans}
    Tuple{SOneTo{1},axes_types(parent_type(T), static(1))}
end
function axes_types(::Type{T}) where {T<:MatAdjTrans}
    Tuple{axes_types(parent_type(T), static(2)),axes_types(parent_type(T), static(1))}
end
function axes_types(::Type{T}) where {T<:PermutedDimsArray}
    eachop_tuple(_get_tuple, to_parent_dims(T), axes_types(parent_type(T)))
end
function axes_types(::Type{T}) where {T<:AbstractRange}
    if known_length(T) === nothing
        return Tuple{OneTo{Int}}
    else
        return Tuple{SOneTo{known_length(T)}}
    end
end
axes_types(::Type{T}) where {T<:ReshapedArray} = NTuple{ndims(T),OneTo{Int}}
function _sub_axis_type(::Type{I}, dim::StaticInt{D}) where {I<:Tuple,D}
    axes_types(_get_tuple(I, dim), static(1))
end
@inline function axes_types(::Type{T}) where {N,P,I,T<:SubArray{<:Any,N,P,I}}
    return eachop_tuple(_sub_axis_type, to_parent_dims(T), I)
end

function axes_types(::Type{T}) where {T<:ReinterpretArray}
    eachop_tuple(_non_reshaped_axis_type, nstatic(Val(ndims(T))), T)
end

function _non_reshaped_axis_type(::Type{A}, d::StaticInt{D}) where {A,D}
    paxis = axes_types(parent_type(A), d)
    if D === 1
        if known_length(paxis) === nothing
            return paxis
        else
            return SOneTo{div(len * sizeof(eltype(parent_type(A))), sizeof(eltyp(A)))}
        end
    else
        return paxis
    end
end

"""
    axes(A) -> Tuple{Vararg{AbstractUnitRange{Int}}}
    axes(A, dim) -> AbstractUnitRange{Int}

Returns the axis associated with each dimension of `A` or dimension `dim`
"""
@inline function axes(a::A) where {A}
    if parent_type(A) <: A
        return Base.axes(a)
    else
        return axes(parent(a))
    end
end
axes(A::ReshapedArray) = Base.axes(A)
axes(A::PermutedDimsArray) = permute(axes(parent(A)), to_parent_dims(A))
axes(A::MatAdjTrans) = permute(axes(parent(A)), to_parent_dims(A))
axes(A::VecAdjTrans) = (SOneTo{1}(), axes(parent(A), 1))
axes(A::SubArray) = map(Base.axes1, permute(A.indices, to_parent_dims(A)))

axes(A::ReshapedArray, dim) = Base.axes(A, Int(dim))
@inline function axes(a::A, dim) where {A}
    d = to_dims(A, dim)
    if parent_type(A) <: A
        if d > ndims(A)
            return SOneTo{1}()
        else
            return Base.axes(a, Int(d))
        end
    else
        return axes(parent(a), to_parent_dims(a, d))
    end
end
axes(A::SubArray, dim) = Base.axes(getindex(A.indices, to_parent_dims(A, to_dims(A, dim))), 1)
if isdefined(Base, :ReshapedReinterpretArray)
    function axes_types(::Type{A}) where {T,N,S,A<:Base.ReshapedReinterpretArray{T,N,S}}
        if sizeof(S) > sizeof(T)
            return merge_tuple_type(Tuple{SOneTo{div(sizeof(S), sizeof(T))}}, axes_types(parent_type(A)))
        elseif sizeof(S) < sizeof(T)
            P = parent_type(A)
            return eachop_tuple(_get_tuple, tail(nstatic(Val(ndims(P)))), axes_types(P))
        else
            return axes_types(parent_type(A))
        end
    end
    @inline function axes(A::Base.ReshapedReinterpretArray{T,N,S}) where {T,N,S}
        if sizeof(S) > sizeof(T)
            return (static(1):static(div(sizeof(S), sizeof(T))), axes(parent(A))...)
        elseif sizeof(S) < sizeof(T)
            return tail(axes(parent(A)))
        else
            return axes(parent(A))
        end
    end
    @inline function axes(A::Base.ReshapedReinterpretArray{T,N,S}, dim) where {T,N,S}
        d = to_dims(A, dim)
        if sizeof(S) > sizeof(T)
            if d == 1
                return static(1):static(div(sizeof(S), sizeof(T)))
            else
                return axes(parent(A), d - static(1))
            end
        elseif sizeof(S) < sizeof(T)
            return axes(parent(A), d - static(1))
        else
            return axes(parent(A), d)
        end
    end
end

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
@inline lazy_axes(x::PermutedDimsArray) = permute(lazy_axes(parent(x)), to_parent_dims(A))
