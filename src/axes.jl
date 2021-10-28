
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
@inline function axes_types(::Type{T}) where {T}
    if parent_type(T) <: T
        return NTuple{ndims(T),OptionallyStaticUnitRange{One,Int}}
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
            return SOneTo{div(known_length(paxis) * sizeof(eltype(parent_type(A))), sizeof(eltype(A)))}
        end
    else
        return paxis
    end
end

# FUTURE NOTE: we avoid  `SOneTo(1)` when `axis(A, dim::Int)``. This is inended to decreases
# breaking changes for this adopting this method to situations where they clearly benefit
# from the propagation of static axes. This creates the somewhat awkward situation of
# conditionally typed (but inferrable) axes. It also means we can't depend on constant
# propagation to preserve statically sized axes. This should probably be addressed before
# merging into Base Julia.
"""
    axes(A) -> Tuple{Vararg{AbstractUnitRange{Int}}}
    axes(A, dim) -> AbstractUnitRange{Int}

Returns the axis associated with each dimension of `A` or dimension `dim`.
`ArrayInterface.axes(::AbstractArray)` behaves nearly identical to `Base.axes` with the
exception of a handful of types replace `Base.OneTo{Int}` with `ArrayInterface.SOneTo`. For
example, the axis along the first dimension of `Transpose{T,<:AbstractVector{T}}` and
`Adjoint{T,<:AbstractVector{T}}` can be represented by `SOneTo(1)`. Similarly,
`Base.ReinterpretArray`'s first axis may be statically sized. 
"""
@inline axes(A) = Base.axes(A)
axes(A::ReshapedArray) = Base.axes(A)
axes(A::PermutedDimsArray) = permute(axes(parent(A)), to_parent_dims(A))
axes(A::MatAdjTrans) = permute(axes(parent(A)), to_parent_dims(A))
axes(A::VecAdjTrans) = (SOneTo{1}(), axes(parent(A), 1))
axes(A::SubArray) = map(Base.axes1, permute(A.indices, to_parent_dims(A)))

@inline axes(A, dim) = _axes(A, to_dims(A, dim))
@inline function _axes(A, dim::Int)
    if dim > ndims(A)
        return OneTo(1)
    else
        return getfield(axes(A), Int(dim))
    end
end
@inline function _axes(A, ::StaticInt{dim}) where {dim}
    if dim > ndims(A)
        return SOneTo{1}()
    else
        return getfield(axes(A), Int(dim))
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
            return (SOneTo(div(sizeof(S), sizeof(T))), axes(parent(A))...)
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
                return SOneTo(div(sizeof(S), sizeof(T)))
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

    function LazyAxis{N}(parent::P) where {N,P}
        N > 0 && return new{N::Int,P}(parent)
        throw_dim_error(parent, N)
    end
    @inline LazyAxis{:}(parent::P) where {P} = new{ifelse(ndims(P) === 1, 1, :),P}(parent)
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
        return SOneTo{known_length(P)}
    end
end

Base.keys(x::LazyAxis) = keys(parent(x))

Base.IndexStyle(::Type{T}) where {T<:LazyAxis} = IndexStyle(parent_type(T))

can_change_size(::Type{LazyAxis{N,P}}) where {N,P} = can_change_size(P)

known_first(::Type{LazyAxis{N,P}}) where {N,P} = known_offsets(P, static(N))
known_first(::Type{LazyAxis{:,P}}) where {P} = 1
@inline function Base.first(x::LazyAxis{N})::Int where {N}
    if known_first(x) === nothing
        return Int(offsets(parent(x), static(N)))
    else
        return Int(known_first(x))
    end
end
@inline function Base.first(x::LazyAxis{:})::Int
    if known_first(x) === nothing
        return firstindex(parent(x))
    else
        return known_first(x)
    end
end
known_last(::Type{LazyAxis{N,P}}) where {N,P} = known_last(axes_types(P, static(N)))
known_last(::Type{LazyAxis{:,P}}) where {P} = known_length(P)
Base.last(x::LazyAxis) = _last(known_last(x), x)
_last(::Nothing, x) = last(parent(x))
_last(N::Int, x) = N

known_length(::Type{LazyAxis{N,P}}) where {N,P} = known_size(P, static(N))
known_length(::Type{LazyAxis{:,P}}) where {P} = known_length(P)
@inline function Base.length(x::LazyAxis{N})::Int where {N}
    if known_length(x) === nothing
        return size(parent(x), static(N))
    else
        return known_length(x)
    end
end
@inline function Base.length(x::LazyAxis{:})::Int
    if known_length(x) === nothing
        return length(parent(x))
    else
        return known_length(x)
    end
end

Base.axes(x::LazyAxis) = (Base.axes1(x),)
Base.axes1(x::LazyAxis) = x
Base.axes(S::Slice{<:LazyAxis}) = (S.indices,)
Base.axes1(S::Slice{<:LazyAxis}) = S.indices
Base.unsafe_indices(S::Base.Slice{<:LazyAxis}) = (S.indices,)
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
@propagate_inbounds function Base.getindex(x::LazyAxis, s::StepRange{<:Integer})
    @boundscheck checkbounds(x, s)
    range(Int(first(x) + s.start-1), step=Int(step(s)), length=Int(length(s)))
end
@propagate_inbounds Base.getindex(x::LazyAxis, i::AbstractUnitRange{<:Integer}) = parent(x)[i]

Base.show(io::IO, x::LazyAxis{N}) where {N} = print(io, "LazyAxis{$N}($(parent(x))))")

"""
    lazy_axes(x)

Produces a tuple of axes where each axis is constructed lazily. If an axis of `x` is already
constructed or it is simply retrieved.
"""
@generated function lazy_axes(x::X) where {X}
    Expr(:block, Expr(:meta, :inline), Expr(:tuple, [:(LazyAxis{$dim}(x)) for dim in 1:ndims(X)]...))
end
lazy_axes(x::LinearIndices) = axes(x)
lazy_axes(x::CartesianIndices) = axes(x)
@inline lazy_axes(x::MatAdjTrans) = reverse(lazy_axes(parent(x)))
@inline lazy_axes(x::VecAdjTrans) = (SOneTo{1}(), first(lazy_axes(parent(x))))
@inline lazy_axes(x::PermutedDimsArray) = permute(lazy_axes(parent(x)), to_parent_dims(x))
