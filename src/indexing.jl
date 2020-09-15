
_maybe_tail(::Tuple{}) = ()
_maybe_tail(x::Tuple) = tail(x)

#
#    to_indices(A, args)
#
function to_indices end

@propagate_inbounds to_indices(A, args::Tuple) = to_indices(A, axes(A), args)
@propagate_inbounds function to_indices(A, args::Tuple{Any})
    if ndims(A) === 1
        return to_indices(A, axes(A), first(args))
    else
        return to_indices(A, (eachindex(A),), args)
    end
end
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {Arg}
    if argdims(Arg) > 1
        axes_front, axes_tail = IteratorsMD.split(axs, Val(argdims(Arg)))
        return (to_index(axes_front, first(args)), to_indices(A, axes_tail, tail(args))...)
    else
        return (to_index(first(axs), first(args)), to_indices(A, tail(axs), tail(args))...)
    end
end
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{})
    @boundscheck if length(first(axs)) == 1
        throw(BoundsError(first(axs), ()))
    end
    return to_indices(A, tail(axs), args)
end
@propagate_inbounds function to_indices(A, ::Tuple{}, args::Tuple{Arg,Vararg{Any}}) where {Arg}
    return (to_index(Static(1):Static(1), first(args)), to_indices(A, (), tail(args))...)
end
to_indices(A, axs::Tuple{}, args::Tuple{}) = ()

"""
    to_index([::IndexStyle, ]axis, arg) -> index

Convert the argument `arg` that was originally passed to `getindex` for the dimension
corresponding to `axis` into a form for native indexing (`Int`, Vector{Int}, ect). New
axis types with unique behavior should use an `IndexStyle` trait:

```julia
to_index(axis::MyAxisType, arg) = to_index(IndexStyle(axis), axis, arg)
to_index(::MyIndexStyle, axis, arg) = ...
```
"""
@propagate_inbounds to_index(axis, arg) = to_index(IndexStyle(axis), axis, arg)

# Colons get converted to slices by `indices`
to_index(::IndexStyle, axis, ::Colon) = indices(axis)
function to_index(::IndexStyle, axis, arg::Integer)
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return Int(arg)
end
function to_index(::IndexStyle, axis, arg::AbstractArray)
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return arg
end
# used to drop dimensions
to_index(::IndexStyle, axis, arg::CartesianIndices{0}) = arg

"""
    to_axes(A, args, inds)
    to_axes(A, old_axes, args, inds) -> new_axes

Construct new axes given the index arguments `args` and the corresponding `inds`
constructed after `to_indices(A, old_axes, args) -> inds`
"""
@inline to_axes(A, args, inds) = to_axes(A, axes(A), args, inds)
to_axes(A, axs::Tuple{Ax,Vararg{Any}}, args::Tuple{Arg,Vararg{Any}}, inds::Tuple{}) where {Ax,Arg} = ()
@propagate_inbounds function to_axes(A, axs::Tuple{Ax,Vararg{Any}}, args::Tuple{Arg,Vararg{Any}}, inds::Tuple) where {Ax,Arg}
    if argdims(Ax, Arg) === 0
        # drop this dimension
        return to_axes(A, tail(axs), tail(args), tail(inds))
    elseif argdims(Ax, Arg) === 1
        return (to_axis(first(axs), first(args), first(inds)), to_axes(A, tail(axs), tail(args), tail(inds))...)
    else
        # Only multidimensional AbstractArray{Bool} and AbstractVector{CartesianIndex{N}}
        # make it to this point. They collapse several dimensions into one.
        axes_front, axes_tail = IteratorsMD.split(axs, Val(argdims(Arg)))
        return (to_axis(axes_front, first(args)), to_axes(A, axes_tail, tail(args), tail(inds))...)
    end
end

"""
    to_axis(old_axis, arg, index) -> new_axis

Construct an `new_axis` for a newly constructed array that corresponds to the
previously executed `to_index(old_axis, arg) -> index`.
"""
to_axis(axis, arg, inds) = to_axis(IndexStyle(axis), axis, arg, inds)
to_axis(::IndexStyle, axis, arg, inds) = Static(1):static_length(inds)
# TODO Do we need a special pass for handling `to_axis(axs::Tuple, arg, inds)` where `axs`
# are the axes being collapsed?

"""
    argdims(::Type{T}) -> Int

Whats the dimensionality of the indexing argument of type `T`?
"""
argdims(x) = argdims(typeof(x))
# single elements initially map to 1 dimension but are that dimension is subsequently dropped.
argdims(::Type{T}) where {T} = 0
argdims(::Type{T}) where {T<:Colon} = 1
argdims(::Type{T}) where {T<:AbstractArray} = ndims(T)
argdims(::Type{T}) where {N,T<:CartesianIndex{N}} = N
argdims(::Type{T}) where {N,T<:AbstractArray{CartesianIndex{N}}} = N
argdims(::Type{T}) where {N,T<:AbstractArray{<:Any,N}} = N
argdims(::Type{T}) where {N,T<:LogicalIndex{<:Any,<:AbstractArray{Bool,N}}} = N

# TODO ensure this is inferrible
should_flatten(x) = should_flatten(typeof(x))
@generated function should_flatten(::Type{T}) where {T<:Tuple}
    for i in T.parameters
        (argdims(i) > 1) && return true
    end
    return false
end

# `flatten_args` handles the obnoxious indexing arguments passed to `getindex` that
# don't correspond to a single dimension (CartesianIndex, CartesianIndices,
# AbstractArray{Bool}). Splitting this up from `to_indices` has two advantages:
#
# 1. It greatly simplifies `to_indices` so that things like ambiguity errors aren't as
#    likely to occur. It should only occure at the top level of any given call to getindex
#    so it ensures that most internal multidim indexing is less complicated.
# 2. When `to_axes` runs back over the arguments to construct the axes of the new
#    collection all the the indices and args should match up so that less time is
#    wasted on `IteratorsMD.split`.
flatten_args(A, args::Tuple) = flatten_args(A, axes(A), args)
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {Arg}
    return (first(args), flatten_args(A, _maybe_tail(axs), tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:CartesianIndex{N}}
    _, axes_tail = IteratorsMD.split(axs, Val(N))
    return (first(args)..., flatten_args(A, _maybe_tail(axs), tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {Arg<:CartesianIndices{0}}
    return (first(args), flatten_args(A, tail(axs), tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:CartesianIndices{N}}
    _, axes_tail = IteratorsMD.split(axs, Val(N))
    return (first(args)..., flatten_args(A, axes_tail, tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:AbstractArray{N}}
    _, axes_tail = IteratorsMD.split(axs, Val(N))
    return (first(args)..., flatten_args(A, axes_tail, tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:AbstractArray{Bool,N}}
    axes_front, axes_tail = IteratorsMD.split(axs, Val(N))
    if length(args) === 1
        if IndexStyle(A) isa IndexLinear
            return (LogicalIndex{Int}(first(args)),)
        else
            return (LogicalIndex(first(args)),)
        end
    else
        return (LogicalIndex(first(args)), flatten_args(A, axes_tail, tail(args)))
    end
end
flatten_args(A, axs::Tuple, args::Tuple{}) = ()

###
### getindex methods
###
@propagate_inbounds function getindex(A, args...)
    if should_flatten(args)
        return ArrayInterface.getindex(A, flatten_args(A, args)...)
    else
        return unsafe_getindex(A, args, to_indices(A, args))
    end
end

"""
    UnsafeGetIndex <: Function

`UnsafeGetIndex` controls how indices that have been bounds checked and converted to
native axes' indices are used to return the stored values of an array. For example,
if the indices at each dimension are single integers than `UnsafeGetIndex(inds)` returns
`UnsafeGetElement()`. Conversely, if any of the indices are vectors then
`UnsafeGetCollection()` is returned, indicating that a new array needs to be
reconstructed.
"""
abstract type UnsafeGetIndex <: Function end

struct UnsafeGetElement <: UnsafeGetIndex end
const unsafe_get_element = UnsafeGetElement()

struct UnsafeGetCollection <: UnsafeGetIndex end
const unsafe_get_collection = UnsafeGetCollection()

# 1-arg
UnsafeGetIndex(x) = UnsafeGetIndex(typeof(x))
UnsafeGetIndex(x::UnsafeGetElement) = x
UnsafeGetIndex(::Type{T}) where {T<:Integer} = unsafe_get_element
UnsafeGetIndex(::Type{T}) where {T<:AbstractArray} = unsafe_get_collection

# 2-arg
UnsafeGetIndex(x::UnsafeGetIndex, y::UnsafeGetElement) = x
UnsafeGetIndex(x::UnsafeGetElement, y::UnsafeGetIndex) = y
UnsafeGetIndex(x::UnsafeGetElement, y::UnsafeGetElement) = x

# tuple
UnsafeGetIndex(x::Tuple{I}) where {I} = UnsafeGetIndex(I)
@inline function UnsafeGetIndex(x::Tuple{I,Vararg{Any}}) where {I}
    return UnsafeGetIndex(UnsafeGetIndex(I), UnsafeGetIndex(tail(x)))
end

unsafe_getindex(A, args, inds) = UnsafeGetIndex(inds)(A, args, inds)

# These three methods aren't really functional at this point because they need
# to point to the optimized native indexing methods.
unsafe_get_element(A, args, inds) = @inbounds(getindex(parent(A), inds...))

function unsafe_get_collection(A, args, inds)
    return unsafe_reconstruct(
        A,
        @inbounds(getindex(parent(A), inds...)),
        to_axes(A, args, inds)
    )
end

unsafe_reconstruct(A, p, axs) = typeof(A)(p, axs)

