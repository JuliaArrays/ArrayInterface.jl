
"""
    argdims(::IndexStyle, ::Type{T})

Whats the dimensionality of the indexing argument of type `T`?
"""
argdims(A, x) = argdims(IndexStyle(A), typeof(x))
argdims(s::IndexStyle, x) = argdims(s, typeof(x))
# single elements initially map to 1 dimension but that dimension is subsequently dropped.
argdims(::IndexStyle, ::Type{T}) where {T} = 0
argdims(::IndexStyle, ::Type{T}) where {T<:Colon} = 1
argdims(::IndexStyle, ::Type{T}) where {T<:AbstractArray} = ndims(T)
argdims(::IndexStyle, ::Type{T}) where {N,T<:CartesianIndex{N}} = N
argdims(::IndexStyle, ::Type{T}) where {N,T<:AbstractArray{CartesianIndex{N}}} = N
argdims(::IndexStyle, ::Type{T}) where {N,T<:AbstractArray{<:Any,N}} = N
argdims(::IndexStyle, ::Type{T}) where {N,T<:LogicalIndex{<:Any,<:AbstractArray{Bool,N}}} = N
@inline function argdims(s::IndexStyle, ::Type{T}) where {N,T<:Tuple{Vararg{<:Any,N}}}
    return ntuple(i -> argdims(s, T.parameters[i]), Val(N))
end

"""
    flatten_args(A, args::Tuple{Arg,Vararg{Any}}) -> Tuple

This method may be used to flatten out multi-dimensional arguments across several
dimensions prior to performing indexing if any of `args` can be flattened.

See also: [`can_flatten](@ref)

# Extended help

If one wishes to create a new multi-dimensional argument that is altered prior to most of
the indexing pipeline, then it must be supported via the `can_flatten` and a new instance
of `flatten_args`, such as the following:

```julia

function ArrayInterface.flatten_args(A, args::Tuple{Arg,Vararg{Any}}) where {Arg<:NewIndexer}
    return (some_flattening_method(first(args)), flatten_args(s, tail(args))...)
end
```

Note that `A` is _NOT_ specified here. If different methods are necessary to flatten
types of `NewIndexer` and these are known prior to implementation, then they should be
specified using methods called within `flatten_args`:

```julia

function ArrayInterface.flatten_args(A, args::Tuple{Arg,Vararg{Any}}) where {Arg<:NewIndexer}
    return flatten_new_indexer(A, args)
end
flatten_new_indexer(A::Array1, args) = ...
flatten_new_indexer(A::Array2, args) = ...

```
"""
@inline function flatten_args(A, args::Tuple{Arg,Vararg{Any}}) where {Arg}
    return (first(args), flatten_args(A, tail(args))...)
end
@inline function flatten_args(A, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:CartesianIndex{N}}
    return (first(args).I..., flatten_args(A, tail(args))...)
end
@inline function flatten_args(A, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:CartesianIndices{N}}
    return (first(args).indices..., flatten_args(A, tail(args))...)
end
# we preserve CartesianIndices{0} for dropping dimensions
@inline function flatten_args(A, args::Tuple{Arg,Vararg{Any}}) where {Arg<:CartesianIndices{0}}
    return (first(args), flatten_args(A, tail(args))...)
end
@inline function flatten_args(A, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:AbstractArray{Bool,N}}
    if length(args) === 1
        if s isa IndexLinear
            return (LogicalIndex{Int}(first(args)),)
        else
            return (LogicalIndex(first(args)),)
        end
    else
        return (LogicalIndex(first(args)), flatten_args(A, tail(args)))
    end
end
flatten_args(A, args::Tuple{}) = ()


"""
    can_flatten(::Type{A}, ::Type{T}) -> Bool

Returns `true` if an argument passed during indexing can be flattened across multiple
dimensions. For example, `CartesianIndex{N}` can be flattened as a series of `Int`s 
across `N` dimensions. This method is used to trigger `flatten_args` prior to indexing.
If a particular argument-array combination cannot cannot be flattened, then it should be
specified here. Otherwise, `A` should not be specificied when supporting a new
multi-dimensional indexing type. For example, the following is the typical usage:

```julia
ArrayInterface.flatten_args(::Type{A}, ::Type{T}) where {A,T<:NewIndexer} = true
```

but in rare instances this may be necessary:


```julia
ArrayInterface.flatten_args(::Type{A}, ::Type{T}) where {A<:ForbiddenArray,T<:NewIndexer} = false
```

"""
can_flatten(A, x) = can_flatten(typeof(A), typeof(x))
can_flatten(::Type{A}, ::Type{T}) where {A,T} = false
can_flatten(::Type{A}, ::Type{T}) where {A,I<:CartesianIndex,T<:AbstractArray{I}} = false
can_flatten(::Type{A}, ::Type{T}) where {A,T<: CartesianIndices} = true
can_flatten(::Type{A}, ::Type{T}) where {A,N,T<:AbstractArray{Bool,N}} = N > 1
can_flatten(::Type{A}, ::Type{T}) where {A,N,T<:CartesianIndex{N}} = true
@generated function can_flatten(::Type{A}, ::Type{T}) where {A,T<:Tuple}
    for i in T.parameters
        can_flatten(A, i) && return true
    end
    return false
end

"""
    to_indices(A, args::Tuple) -> to_indices(A, axes(A), args)
    to_indices(A, axes::Tuple, args::Tuple)

Maps arguments `args` to the axes of `A`. This is done by iteratively passing each
axis and argument to [`to_index`](@ref). Unique behavior based on the type of `A` may be
accomplished by overloading `to_indices(A, args)`. Unique axis-argument behavior can
be accomplished using `to_index(axis, arg)`.
"""
@propagate_inbounds function to_indices(A, args::Tuple)
    if can_flatten(A, args)
        return to_indices(A, flatten_args(A, args))
    else
        return to_indices(A, axes(A), args)
    end
end
@propagate_inbounds function to_indices(A, args::Tuple{Arg}) where {Arg}
    if can_flatten(A, args)
        return to_indices(A, flatten_args(A, args))
    else
        if argdims(IndexStyle(A), Arg) > 1
            return to_indices(A, axes(A), args)
        else
            if ndims(A) === 1
                return (to_index(axes(A, 1), first(args)),)
            else
                return to_indices(A, (eachindex(A),), args)
            end
        end
    end
end
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {Arg}
    N = argdims(IndexStyle(A), Arg)
    if N > 1
        axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(N))
        return (to_multi_index(axes_front, first(args)), to_indices(A, axes_tail, tail(args))...)
    else
        return (to_index(first(axs), first(args)), to_indices(A, tail(axs), tail(args))...)
    end
end
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{})
    @boundscheck if length(first(axs)) != 1
        error("Cannot drop dimension of size $(length(first(axs))).")
    end
    return to_indices(A, tail(axs), args)
end
@propagate_inbounds function to_indices(A, ::Tuple{}, args::Tuple{Arg,Vararg{Any}}) where {Arg}
    return (to_index(OneTo(1), first(args)), to_indices(A, (), tail(args))...)
end
to_indices(A, axs::Tuple{}, args::Tuple{}) = ()

function to_multi_index(axs::Tuple, arg)
    @boundscheck if !Base.checkbounds_indices(Bool, axs, (arg,))
        throw(BoundsError(axs, arg))
    end
    return arg
end

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
to_index(axis, arg::CartesianIndices{0}) = arg
# Colons get converted to slices by `indices`
to_index(::IndexStyle, axis, ::Colon) = indices(axis)
function to_index(::IndexStyle, axis, arg::Integer)
    @boundscheck checkbounds(axis, arg)
    return Int(arg)
end
@propagate_inbounds function to_index(::IndexStyle, axis, arg::AbstractArray{Bool})
    @boundscheck checkbounds(axis, arg)
    return AbstractArray{Int}(@inbounds(axis[arg]))
end
function to_index(::IndexStyle, axis, arg::AbstractArray{I}) where {I<:Integer}
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return AbstractArray{Int}(arg)
end
@propagate_inbounds function to_index(::IndexStyle, axis, arg::AbstractRange{I}) where {I<:Integer}
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return arg
end
function to_index(S::IndexStyle, axis, arg)
    throw(ArgumentError("invalid index: IndexStyle $S does not support indices of type $(typeof(arg))."))
end

"""
    unsafe_reconstruct(A, data; kwargs...)

Reconstruct `A` given the values in `data`. New methods using `unsafe_reconstruct`
should only dispatch on `A`.
"""
function unsafe_reconstruct(A::AbstractUnitRange, data; kwargs...)
    if can_change_size(A)
        return typeof(A)(data)
    else
        if data isa Slice || !(known_length(A) === nothing || known_length(A) !== known_length(data))
            return A
        else
            return typeof(A)(data)
        end
    end
end

"""
    to_axes(A, inds)
    to_axes(A, old_axes, inds) -> new_axes

Construct new axes given the corresponding `inds` constructed after
`to_indices(A, old_axes, args) -> inds`. This method iterates through each
pair of axes and indices, calling [`to_axis`](@ref).
"""
@inline function to_axes(A, inds::Tuple)
    if ndims(A) === 1
        return (to_axis(axes(A, 1), first(inds)),)
    else
        return to_axes(A, axes(A), inds)
    end
end
to_axes(A, ::Tuple{Ax,Vararg{Any}}, ::Tuple{}) where {Ax} = ()
to_axes(A, ::Tuple{}, ::Tuple{}) = ()
@propagate_inbounds function to_axes(A, axs::Tuple{Ax,Vararg{Any}}, inds::Tuple{I,Vararg{Any}}) where {Ax,I}
    N = argdims(IndexStyle(A), I)
    if N === 0
        # drop this dimension
        return to_axes(A, tail(axs), tail(inds))
    elseif N === 1
        return (to_axis(first(axs), first(inds)), to_axes(A, tail(axs), tail(inds))...)
    else
        # Only multidimensional AbstractArray{Bool} and AbstractVector{CartesianIndex{N}}
        # make it to this point. They collapse several dimensions into one.
        axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(N))
        return (to_multi_axis(IndexStyle(A), axes_front, first(inds)), to_axes(A, axes_tail, tail(inds))...)
    end
end

"""
    to_axis(old_axis, index) -> new_axis

Construct an `new_axis` for a newly constructed array that corresponds to the
previously executed `to_index(old_axis, arg) -> index`. `to_axis` assumes that
`index` has already been confirmed to be inbounds. The underlying indices of
`new_axis` begins at one and extends the length of `index` (i.e. one-based indexing).
"""
@inline function to_axis(axis, inds)
    if !can_change_size(axis) && (known_length(inds) !== nothing && known_length(axis) === known_length(inds))
        return axis
    else
        return to_axis(IndexStyle(axis), axis, inds)
    end
end
@inline function to_axis(S::IndexStyle, axis, inds)
    return unsafe_reconstruct(axis, StaticInt(1):static_length(inds))
end
@inline function to_multi_axis(::IndexStyle, axs::Tuple, inds)
    return to_axis(eachindex(LinearIndices(axs)), inds)
end

"""
    ArrayInterface.getindex(A, args...)

Retrieve the value(s) stored at the given key or index within a collection. Creating
other instance of `ArrayInterface.getindex` should only be done by overloading `A`.
Changing indexing based on a given argument from `args` should be done through
[`flatten_args`](@ref), [`to_index`](@ref), or [`to_axis`](@ref).
"""
@propagate_inbounds getindex(A, args...) = unsafe_getindex(A, to_indices(A, args))

"""
    UnsafeIndex <: Function

`UnsafeIndex` controls how indices that have been bounds checked and converted to
native axes' indices are used to return the stored values of an array. For example,
if the indices at each dimension are single integers than `UnsafeIndex(inds)` returns
`UnsafeElement()`. Conversely, if any of the indices are vectors then `UnsafeCollection()`
is returned, indicating that a new array needs to be reconstructed. This method permits
customizing the terimnal behavior of the indexing pipeline based on arguments passed
to `ArrayInterface.getindex`
"""
abstract type UnsafeIndex <: Function end

struct UnsafeElement <: UnsafeIndex end
const unsafe_element = UnsafeElement()

struct UnsafeCollection <: UnsafeIndex end
const unsafe_collection = UnsafeCollection()

# 1-arg
UnsafeIndex(x) = UnsafeIndex(typeof(x))
UnsafeIndex(x::UnsafeIndex) = x
UnsafeIndex(::Type{T}) where {T<:Integer} = unsafe_element
UnsafeIndex(::Type{T}) where {T<:AbstractArray} = unsafe_collection

# 2-arg
UnsafeIndex(x::UnsafeIndex, y::UnsafeElement) = x
UnsafeIndex(x::UnsafeElement, y::UnsafeIndex) = y
UnsafeIndex(x::UnsafeElement, y::UnsafeElement) = x
UnsafeIndex(x::UnsafeCollection, y::UnsafeCollection) = x


# tuple
UnsafeIndex(x::Tuple{I}) where {I} = UnsafeIndex(I)
@inline function UnsafeIndex(x::Tuple{I,Vararg{Any}}) where {I}
    return UnsafeIndex(UnsafeIndex(I), UnsafeIndex(tail(x)))
end

"""
    unsafe_getindex(A, inds)

Indexes into `A` given `inds`. This method assumes that `inds` have already been
bounds checked.
"""
unsafe_getindex(A, inds) = unsafe_getindex(UnsafeIndex(inds), A, inds)
unsafe_getindex(::UnsafeElement, A, inds) = unsafe_get_element(A, inds)
unsafe_getindex(::UnsafeCollection, A, inds) = unsafe_get_collection(A, inds)

"""
    unsafe_get_element(A::AbstractArray{T}, inds::Tuple) -> T

Returns an element of `A` at the indices `inds`. This method assumes all `inds`
have been checked for being inbounds. Any new array type using `ArrayInterface.getindex`
must define `unsafe_get_element(::NewArrayType, inds)`
"""
function unsafe_get_element(A, inds)
    throw(MethodError(unsafe_getindex, (A, inds)))
end
function unsafe_get_element(A::Array, inds)
    if inds isa Tuple{Vararg{Int}}
        return Base.arrayref(false, A, inds...)
    else
        throw(MethodError(unsafe_get_element, (A, inds)))
    end
end
@inline function unsafe_get_element(A::LinearIndices, inds)
    return Int(Base._to_linear_index(A, inds...))
end
@inline function unsafe_get_element(A::CartesianIndices, inds)
    return CartesianIndex(Base._to_subscript_indices(A, inds...))
end

# This is based on Base._unsafe_getindex from https://github.com/JuliaLang/julia/blob/c5ede45829bf8eb09f2145bfd6f089459d77b2b1/base/multidimensional.jl#L755
"""
    unsafe_get_collection(A, inds)

Returns a collection of `A` given `inds`. `inds` is assumed to be bounds checked prior.
"""
function unsafe_get_collection(A, inds)
    axs = to_axes(A, inds)
    dest = similar(A, axs)
    map(Base.unsafe_length, axes(dest)) == map(Base.unsafe_length, axs) || throw_checksize_error(dest, axs)
    Base._unsafe_getindex!(dest, A, inds...) # usually a generated function, don't allow it to impact inference result
    return dest
end

can_preserve_indices(::Type{T}) where {T<:AbstractRange} = known_step(T) === 1
can_preserve_indices(::Type{T}) where {T<:Int} = true
can_preserve_indices(::Type{T}) where {T} = false

ints2range(x::Integer) = x:x
ints2range(x::AbstractRange) = x

# if linear indexing on multidim or can't reconstruct AbstractUnitRange
# then contstruct Array of CartesianIndex/LinearIndices
@generated function can_preserve_indices(::Type{T}) where {T<:Tuple}
    for index_type in T.parameters
        can_preserve_indices(index_type) || return false
    end
    return true
end

@inline function unsafe_get_collection(A::CartesianIndices{N}, inds) where {N}
    if (length(inds) === 1 && N > 1) || !can_preserve_indices(typeof(inds))
        return Base._getindex(IndexStyle(A), A, inds...)
    else
        return CartesianIndices(to_axes(A, ints2range.(inds)))
    end
end
@inline function unsafe_get_collection(A::LinearIndices{N}, inds) where {N}
    if can_preserve_indices(typeof(inds))
        return LinearIndices(to_axes(A, ints2range.(inds)))
    else
        if length(inds) === 1
            return @inbounds(eachindex(A)[first(inds)])
        else
            return Base._getindex(IndexStyle(A), A, inds...)
        end
    end
end

"""
    ArrayInterface.setindex!(A, args...)

Store the given values at the given key or index within a collection.
"""
@propagate_inbounds function setindex!(A, val, args...)
    if can_setindex(A)
        return unsafe_setindex!(A, val, to_indices(A, args))
    else
        error("Instance of type $(typeof(A)) are not mutable and cannot change " *
              "elements after construction.")
    end
end

"""
    unsafe_setindex!(A, val, inds::Tuple)

Sets indices (`inds`) of `A` to `val`. This method assumes that `inds` have already been
bounds checked. This step of the processing pipeline can be customized by
"""
unsafe_setindex!(A, val, inds::Tuple) = unsafe_setindex!(UnsafeIndex(inds), A, val, inds)
unsafe_setindex!(::UnsafeElement, A, val, inds::Tuple) = unsafe_set_element!(A, val, inds)
unsafe_setindex!(::UnsafeCollection, A, val, inds::Tuple) = unsafe_set_collection!(A, val, inds)

"""
    unsafe_set_element!(A, val, inds::Tuple)

Sets an element of `A` to `val` at indices `inds`. This method assumes all `inds`
have been checked for being inbounds. Any new array type using `ArrayInterface.setindex!`
must define `unsafe_set_element!(::NewArrayType, val, inds)`.
"""
function unsafe_set_element!(A, val, inds)
    throw(MethodError(unsafe_set_element!, (A, val, inds)))
end
function unsafe_set_element!(A::Array{T}, val, inds::Tuple) where {T}
    if inds isa Tuple{Vararg{Int}}
        return Base.arrayset(false, A, convert(T, val)::T, inds...)
    else
        throw(MethodError(unsafe_set_element!, (A, inds)))
    end
end

# This is based on Base._unsafe_setindex!
"""
    unsafe_set_collection!(A, val, inds)

Sets `inds` of `A` to `val`. `inds` is assumed to be bounds checked prior.
"""
@inline function unsafe_set_collection!(A, val, inds)
    return Base._unsafe_setindex!(IndexStyle(A), A, val, inds...)
end

