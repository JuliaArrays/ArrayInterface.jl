
""" can_flatten(::Type{T}) """ # TODO document new flatten
@inline flatten(A, inds) = _flatten(can_flatten(inds), A, inds)
@inline _flatten(::True, A, inds) = flatten_indices(A, inds)
_flatten(::False, A, inds) = inds

can_flatten(x) = can_flatten(typeof(x))
can_flatten(::Type{T}) where {T} = static(false)
can_flatten(::Type{T}) where {T<:AbstractArray{<:AbstractCartesianIndex}} = static(false)
can_flatten(::Type{T}) where {T<:CartesianIndices} = static(true)
can_flatten(::Type{T}) where {T<:AbstractArray{Bool}} = static(ndims(T) > 1)
can_flatten(::Type{T}) where {T<:AbstractCartesianIndex} = static(true)

can_flatten(::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}} = _can_flatten(T, static(N))
_can_flatten(::Type{T}, ::StaticInt{0}) where {T} = static(false)
@inline function _can_flatten(::Type{T}, n::StaticInt{N}) where {T,N}
    return can_flatten(_get_tuple(T, n)) | _can_flatten(T, n - static(1))
end

"""
    flatten_indices(A, inds) -> flatten_indices(axes(A), inds)

Flatten multi-dimension spanning argument into separate arguments for each dimension.
"""
flatten_indices(A, i::Tuple) = flatten_indices(lazy_axes(A), i)
@inline function flatten_indices(a::Tuple, i::Tuple{I,Vararg{Any}}) where {I}
    return (first(i), flatten_indices(tail(a), tail(i))...)
end
@inline function flatten_indices(a::Tuple, i::Tuple{I,Vararg{Any}}) where {N,I<:AbstractCartesianIndex{N}}
    _, atail = Base.IteratorsMD.split(a, Val(N))
    return (Tuple(first(i))..., flatten_indices(atail, tail(i))...)
end
@inline function flatten_indices(a::Tuple, i::Tuple{I,Vararg{Any}}) where {N,I<:CartesianIndices{N}}
    _, atail = Base.IteratorsMD.split(a, Val(N))
    return (axes(first(i))..., flatten_indices(atail, tail(i))...)
end
# we preserve CartesianIndices{0} for dropping dimensions
@inline function flatten_indices(a::Tuple, i::Tuple{I,Vararg{Any}}) where {I<:CartesianIndices{0}}
    return (first(i), flatten_indices(tail(a), tail(i))...)
end
flatten_indices(::Tuple, ::Tuple{}) = ()

## is_canonical ##
is_canonical(x) = is_canonical(typeof(x))
is_canonical(::Type{T}) where {T} = static(false)
is_canonical(::Type{Int}) = static(true)
is_canonical(::Type{<:StaticInt}) = static(true)
is_canonical(::Type{StepRange{Int,Int}}) = static(true)
is_canonical(::Type{UnitRange{Int}}) = static(true)
is_canonical(::Type{OneTo{Int}}) = static(true)
is_canonical(::Type{<:OptionallyStaticRange}) = static(true)
is_canonical(::Type{Vector{Int}}) = static(true)
is_canonical(::Type{<:Slice}) = static(true)
@inline is_canonical(::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}} = _is_canon(T, static(N))
_is_canon(::Type{T}, ::StaticInt{0}) where {T} = static(true)
@inline function _is_canon(::Type{T}, n::StaticInt{N}) where {T,N}
    return is_canonical(_get_tuple(T, n)) & _is_canon(T, n - static(1))
end

"""
    canonicalize(x)

Checks if `x` is in a canonical form for indexing. If `x` is already in a canonical form
then it is returned unchanged. If `x` is not in a canonical form then it is passed to
`canonical_convert`.
"""
canonicalize(x) = _canonicalize(is_canonical(x), x)
_canonicalize(::True, x) = x
_canonicalize(::False, x) = canonical_convert(x)

canonical_convert(x::Integer) = convert(Int, x)
function canonical_convert(x::AbstractRange)
    return OptionallyStaticStepRange(static_first(x), static_step(x), static_last(x))
end
function canonicalize_convert(x::AbstractUnitRange{<:Integer})
    return OptionallyStaticUnitRange(static_first(x), static_last(x))
end

_layout(::IndexLinear, x::Tuple) = LinearIndices(x)
_layout(::IndexCartesian, x::Tuple) = CartesianIndices(x)

"""
    ArrayStyle(::Type{A})

Used to customize the meaning of indexing arguments in the context of a given array `A`.

See also: [`argdims`](@ref), [`UnsafeIndex`](@ref)
"""
abstract type ArrayStyle end

struct DefaultArrayStyle <: ArrayStyle end

ArrayStyle(A) = ArrayStyle(typeof(A))
ArrayStyle(::Type{A}) where {A} = DefaultArrayStyle()

"""
    argdims(::ArrayStyle, ::Type{T})

What is the dimensionality of the indexing argument of type `T`?
"""
argdims(x, arg) = argdims(x, typeof(arg))
argdims(x, ::Type{T}) where {T} = argdims(ArrayStyle(x), T)
argdims(s::ArrayStyle, arg) = argdims(s, typeof(arg))
# single elements initially map to 1 dimension but that dimension is subsequently dropped.
argdims(::ArrayStyle, ::Type{T}) where {T} = static(0)
argdims(::ArrayStyle, ::Type{T}) where {T<:Colon} = static(1)
argdims(::ArrayStyle, ::Type{T}) where {T<:AbstractArray} = static(ndims(T))
argdims(::ArrayStyle, ::Type{T}) where {N,T<:AbstractCartesianIndex{N}} = static(N)
argdims(::ArrayStyle, ::Type{T}) where {N,T<:AbstractArray{<:AbstractCartesianIndex{N}}} = static(N)
argdims(::ArrayStyle, ::Type{T}) where {N,T<:AbstractArray{<:Any,N}} = static(N)
argdims(::ArrayStyle, ::Type{T}) where {N,T<:LogicalIndex{<:Any,<:AbstractArray{Bool,N}}} = static(N)
_argdims(s::ArrayStyle, ::Type{I}, i::StaticInt) where {I} = argdims(s, _get_tuple(I, i))
function argdims(s::ArrayStyle, ::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}}
    return eachop(_argdims, nstatic(Val(N)), s, T)
end

_is_element_index(i) = _is_element_index(typeof(i))
_is_element_index(::Type{T}) where {T} = static(false)
_is_element_index(::Type{T}) where {T<:AbstractCartesianIndex} = static(true)
_is_element_index(::Type{T}) where {T<:Integer} = static(true)
__is_element_index(::Type{T}, i::StaticInt) where {T} = _is_element_index(_get_tuple(T, i))
function _is_element_index(::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}}
    return static(all(eachop(__is_element_index, nstatic(Val(N)), T)))
end
# empty tuples refer to the single element of 0-dimensional arrays
_is_element_index(::Type{Tuple{}}) = static(true)

# are the indexing arguments provided a linear collection into a multidim collection
is_linear_indexing(A, args::Tuple{Arg}) where {Arg} = argdims(A, Arg) < 2
is_linear_indexing(A, args::Tuple{Arg,Vararg{Any}}) where {Arg} = false

"""
    to_indices(A, args::Tuple) -> to_indices(A, axes(A), args)
    to_indices(A, axes::Tuple, args::Tuple)

Maps arguments `args` to the axes of `A`. This is done by iteratively passing each
axis and argument to [`to_index`](@ref). Unique behavior based on the type of `A` may be
accomplished by overloading `to_indices(A, args)`. Unique axis-argument behavior can
be accomplished using `to_index(axis, arg)`.
"""
@propagate_inbounds to_indices(A, args::Tuple) = _to_indices(A, flatten(A, args))
@propagate_inbounds function to_indices(A, args::Tuple{LinearIndices})
    return to_indices(A, lazy_axes(A), axes(first(args)))
end
@propagate_inbounds _to_indices(A, args::Tuple) = __to_indices(is_canonical(args), A, args)
@propagate_inbounds function __to_indices(::False, A, args)
    if is_linear_indexing(A, args)
        return (to_index(eachindex(IndexLinear(), A), first(args)),)
    else
        return to_indices(A, lazy_axes(A), args)
    end
end
# don't waste time reconstructing tuple if already converted
@propagate_inbounds function __to_indices(::True, A, args)
    if is_linear_indexing(A, args)
        @boundscheck if !checkindex(Bool, eachindex(IndexLinear(), A), first(args))
            throw(BoundsError(A, args))
        end
        return args
    else
        @boundscheck if !Base.checkbounds_indices(Bool, lazy_axes(A), args)
            throw(BoundsError(A, args))
        end
        return args
    end
end

@propagate_inbounds to_indices(A, args::Tuple{}) = to_indices(A, lazy_axes(A), ())
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{I,Vararg{Any}},) where {I}
    return _to_indices(argdims(A, I), A, axs, args)
end
@propagate_inbounds function _to_indices(::StaticInt{0}, A, axs::Tuple, args::Tuple)
    return (to_index(first(axs), first(args)), to_indices(A, _maybe_tail(axs), _maybe_tail(args))...)
end
@propagate_inbounds function _to_indices(::StaticInt{1}, A, axs::Tuple, args::Tuple)
    return (to_index(first(axs), first(args)), to_indices(A, _maybe_tail(axs), _maybe_tail(args))...)
end
@propagate_inbounds function _to_indices(::StaticInt{N}, A, axs::Tuple, args::Tuple) where {N}
    axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    return (to_index(_layout(IndexStyle(A), axes_front), first(args)), to_indices(A, axes_tail, _maybe_tail(args))...)
end
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{})
    @boundscheck if length(first(axs)) != 1
        error("Cannot drop dimension of size $(length(first(axs))).")
    end
    return to_indices(A, tail(axs), args)
end
@propagate_inbounds function to_indices(A, ::Tuple{}, args::Tuple{I,Vararg{Any}}) where {I}
    return (to_index(OneTo(1), first(args)), to_indices(A, (), tail(args))...)
end
to_indices(A, axs::Tuple{}, args::Tuple{}) = ()
_maybe_tail(::Tuple{}) = ()
_maybe_tail(x::Tuple) = tail(x)

"""
    to_index([::IndexStyle, ]axis, arg) -> index

Convert the argument `arg` that was originally passed to `getindex` for the dimension
corresponding to `axis` into a form for native indexing (`Int`, Vector{Int}, etc.). New
axis types with unique behavior should use an `IndexStyle` trait:

```julia
to_index(axis::MyAxisType, arg) = to_index(IndexStyle(axis), axis, arg)
to_index(::MyIndexStyle, axis, arg) = ...
```
"""
@propagate_inbounds to_index(axis, arg) = to_index(IndexStyle(axis), axis, arg)
function to_index(s, axis, arg)
    throw(ArgumentError("invalid index: IndexStyle $s does not support indices of " *
                        "type $(typeof(arg)) for instances of type $(typeof(axis))."))
end
to_index(::IndexLinear, axis, arg::Colon) = indices(axis) # Colons get converted to slices by `indices`
to_index(::IndexLinear, axis, arg::CartesianIndices{0}) = arg
to_index(::IndexLinear, axis, arg::CartesianIndices{1}) = axes(arg, 1)
@propagate_inbounds function to_index(::IndexLinear, axis, arg::AbstractCartesianIndex{1})
    return to_index(axis, first(Tuple(arg)))
end
function to_index(::IndexLinear, x, arg::AbstractCartesianIndex{N}) where {N}
    inds = Tuple(arg)
    o = offsets(x)
    s = size(x)
    return first(inds) + (offset1(x) - first(o)) + _subs2int(first(s), tail(s), tail(o), tail(inds))
end
@inline function _subs2int(stride, s::Tuple{Any,Vararg}, o::Tuple{Any,Vararg}, inds::Tuple{Any,Vararg})
    i = ((first(inds) - first(o)) * stride)
    return i + _subs2int(stride * first(s), tail(s), tail(o), tail(inds))
end
function _subs2int(stride, s::Tuple{Any}, o::Tuple{Any}, inds::Tuple{Any})
    return (first(inds) - first(o)) * stride
end
# trailing inbounds can only be 1 or 1:1
_subs2int(stride, ::Tuple{}, ::Tuple{}, ::Tuple{Any}) = static(0)

@propagate_inbounds function to_index(::IndexLinear, x, arg::Union{Array{Bool}, BitArray})
    @boundscheck checkbounds(x, arg)
    return LogicalIndex{Int}(arg)
end
@propagate_inbounds function to_index(::IndexLinear, x, arg::AbstractArray{<:AbstractCartesianIndex})
    @boundscheck _multi_check_index(axes(x), arg) || throw(BoundsError(x, arg))
    return arg
end
@propagate_inbounds function to_index(::IndexLinear, x, arg::LogicalIndex)
    @boundscheck checkbounds(Bool, x, arg) || throw(BoundsError(x, arg))
    return arg
end
@propagate_inbounds function to_index(::IndexLinear, x, arg::Integer)
    @boundscheck checkindex(Bool, indices(x), arg) || throw(BoundsError(x, arg))
    return canonicalize(arg)
end
@propagate_inbounds function to_index(::IndexLinear, axis, arg::AbstractArray{Bool})
    @boundscheck checkbounds(axis, arg)
    return LogicalIndex(arg)
end
@propagate_inbounds function to_index(::IndexLinear, x, arg::AbstractArray{<:Integer})
    @boundscheck checkindex(Bool, x, arg) || throw(BoundsError(x, arg))
    return arg
end
@propagate_inbounds function to_index(::IndexLinear, x, arg::AbstractRange{Integer})
    @boundscheck checkindex(Bool, indices(axis), arg) || throw(BoundsError(axis, arg))
    return static_first(arg):static_step(arg):static_last(arg)
end

## IndexCartesian ##
to_index(::IndexCartesian, x, arg::Colon) = CartesianIndices(x)
to_index(::IndexCartesian, x, arg::CartesianIndices{0}) = arg
to_index(::IndexCartesian, x, arg::AbstractCartesianIndex) = arg
function to_index(::IndexCartesian, x, arg)
    @boundscheck _multi_check_index(axes(x), arg) || throw(BoundsError(x, arg))
    return arg
end
@propagate_inbounds function to_index(::IndexCartesian, x, arg::AbstractArray{<:AbstractCartesianIndex})
    @boundscheck _multi_check_index(axes(x), arg) || throw(BoundsError(x, arg))
    return arg
end
@propagate_inbounds function to_index(::IndexCartesian, x, arg::AbstractArray{Bool})
    @boundscheck checkbounds(x, arg)
    return LogicalIndex(arg)
end

function _multi_check_index(axs::Tuple, arg::AbstractArray{T}) where {T<:AbstractCartesianIndex}
    b = true
    for i in arg
        b &= Base.checkbounds_indices(Bool, axs, (i,))
    end
    return b
end

@propagate_inbounds function to_index(::IndexCartesian, x, arg::Union{Array{Bool}, BitArray})
    @boundscheck checkbounds(x, arg)
    return LogicalIndex{Int}(arg)
end
to_index(::IndexCartesian, x, i::Integer) = NDIndex(_int2subs(offsets(x), size(x), i - offset1(x)))
@inline function _int2subs(o::Tuple{Any,Vararg{Any}}, s::Tuple{Any,Vararg{Any}}, i)
    len = first(s)
    inext = div(i, len)
    return (canonicalize(i - len * inext + first(o)), _int2subs(tail(o), tail(s), inext)...)
end
_int2subs(o::Tuple{Any}, s::Tuple{Any}, i) = canonicalize(i + first(o))

"""
    unsafe_reconstruct(A, data; kwargs...)

Reconstruct `A` given the values in `data`. New methods using `unsafe_reconstruct`
should only dispatch on `A`.
"""
function unsafe_reconstruct(axis::OneTo, data; kwargs...)
    if axis === data
        return axis
    else
        return OneTo(data)
    end
end
function unsafe_reconstruct(axis::UnitRange, data; kwargs...)
    if axis === data
        return axis
    else
        return UnitRange(first(data), last(data))
    end
end
function unsafe_reconstruct(axis::OptionallyStaticUnitRange, data; kwargs...)
    if axis === data
        return axis
    else
        return OptionallyStaticUnitRange(static_first(data), static_last(data))
    end
end
function unsafe_reconstruct(A::AbstractUnitRange, data; kwargs...)
    return static_first(data):static_last(data)
end

"""
    to_axes(A, inds)

Construct new axes given the corresponding `inds` constructed after
`to_indices(A, args) -> inds`. This method iterates through each pair of axes and
indices calling [`to_axis`](@ref).
"""
@inline function to_axes(A, inds::Tuple)
    if ndims(A) === 1
        return (to_axis(axes(A, 1), first(inds)),)
    elseif is_linear_indexing(A, inds)
        return (to_axis(eachindex(IndexLinear(), A), first(inds)),)
    else
        return to_axes(A, axes(A), inds)
    end
end
to_axes(A, a::Tuple, i::Tuple{I,Vararg{Any}},) where {I} = _to_axes(argdims(A, I), A, a, i)
# drop this dimension
_to_axes(::StaticInt{0}, A, axs::Tuple, inds::Tuple) = to_axes(A, tail(inds), tail(inds))
function _to_axes(::StaticInt{1}, A, axs::Tuple, inds::Tuple)
    return (to_axis(first(axs), first(inds)), to_axes(A, tail(axs), tail(inds))...)
end
@propagate_inbounds function _to_axes(::StaticInt{N}, A, axs::Tuple, inds::Tuple) where {N}
    axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    return (
        to_axis(_layout(IndexStyle(A), axes_front), first(inds)),
        to_axes(A, axes_tail, tail(inds))...
    )
end
to_axes(A, ::Tuple{Ax,Vararg{Any}}, ::Tuple{}) where {Ax} = ()
to_axes(A, ::Tuple{}, ::Tuple{}) = ()

"""
    to_axis(old_axis, index) -> new_axis

Construct an `new_axis` for a newly constructed array that corresponds to the
previously executed `to_index(old_axis, arg) -> index`. `to_axis` assumes that
`index` has already been confirmed to be in bounds. The underlying indices of
`new_axis` begins at one and extends the length of `index` (i.e., one-based indexing).
"""
@inline function to_axis(axis, inds)
    if !can_change_size(axis) &&
       (known_length(inds) !== nothing && known_length(axis) === known_length(inds))
        return axis
    else
        return to_axis(IndexStyle(axis), axis, inds)
    end
end

# don't need to check size b/c slice means it's the entire axis
@inline function to_axis(axis, inds::Slice)
    if can_change_size(axis)
        return copy(axis)
    else
        return axis
    end
end
to_axis(S::IndexLinear, axis, inds) = StaticInt(1):static_length(inds)

################
### getindex ###
################
"""
    ArrayInterface.getindex(A, args...)

Retrieve the value(s) stored at the given key or index within a collection. Creating
another instance of `ArrayInterface.getindex` should only be done by overloading `A`.
Changing indexing based on a given argument from `args` should be done through,
[`to_index`](@ref), or [`to_axis`](@ref).
"""
@propagate_inbounds getindex(A, args...) = unsafe_get_index(A, to_indices(A, args))
@propagate_inbounds function getindex(A; kwargs...)
    return unsafe_get_index(A, to_indices(A, order_named_inds(dimnames(A), kwargs.data)))
end
@propagate_inbounds getindex(x::Tuple, i::Int) = getfield(x, i)
@propagate_inbounds getindex(x::Tuple, ::StaticInt{i}) where {i} = getfield(x, i)

## unsafe_get_index ##
unsafe_get_index(A, inds::Tuple) = _unsafe_get_index(_is_element_index(inds), A, inds)
_unsafe_get_index(::False, A, inds::Tuple) = unsafe_get_collection(A, inds)
_unsafe_get_index(::True, A, inds::Tuple) = __unsafe_get_index(A, inds)
__unsafe_get_index(A, inds::Tuple{}) = unsafe_get_element(A, ())
__unsafe_get_index(A, inds::Tuple{Any}) = unsafe_get_element(A, first(inds))
__unsafe_get_index(A, inds::Tuple{Any,Vararg{Any}}) = unsafe_get_element(A, NDIndex(inds))

"""
    unsafe_get_element(A::AbstractArray{T}, inds::Tuple) -> T

Returns an element of `A` at the indices `inds`. This method assumes all `inds`
have been checked for being in bounds. Any new array type using `ArrayInterface.getindex`
must define `unsafe_get_element(::NewArrayType, inds)`.
"""
unsafe_get_element(a::A, inds) where {A} = _unsafe_get_element(has_parent(A), a, inds)
_unsafe_get_element(::True, a, inds) = unsafe_get_element(parent(a), inds)
_unsafe_get_element(::False, a, inds) = @inbounds(parent(a)[inds])
_unsafe_get_element(::False, a::AbstractArray2, i) = unsafe_get_element_error(a, i)

## Array ##
unsafe_get_element(A::Array, ::Tuple{}) = Base.arrayref(false, A, 1)
unsafe_get_element(A::Array, i::Integer) = Base.arrayref(false, A, Int(i))
unsafe_get_element(A::Array, i::NDIndex) = unsafe_get_element(A, to_index(A, i))

## LinearIndices ##
unsafe_get_element(A::LinearIndices, i::Integer) = Int(i)
unsafe_get_element(A::LinearIndices, i::NDIndex) = unsafe_get_element(A, to_index(A, i))

unsafe_get_element(A::CartesianIndices, i::NDIndex) = CartesianIndex(i)
unsafe_get_element(A::CartesianIndices, i::Integer) = unsafe_get_element(A, to_index(A, i))

unsafe_get_element(A::ReshapedArray, i::Integer) = unsafe_get_element(parent(A), i)
function unsafe_get_element(A::ReshapedArray, i::NDIndex)
    return unsafe_get_element(parent(A), to_index(IndexLinear(), A, i))
end

unsafe_get_element(A::SubArray, i) = @inbounds(A[i])
function unsafe_get_element_error(@nospecialize(A), @nospecialize(i))
    throw(MethodError(unsafe_get_element, (A, i)))
end

# This is based on Base._unsafe_getindex from https://github.com/JuliaLang/julia/blob/c5ede45829bf8eb09f2145bfd6f089459d77b2b1/base/multidimensional.jl#L755.
"""
    unsafe_get_collection(A, inds)

Returns a collection of `A` given `inds`. `inds` is assumed to have been bounds-checked.
"""
function unsafe_get_collection(A, inds)
    axs = to_axes(A, inds)
    dest = similar(A, axs)
    unsafe_get_index!(layout(A, AccessStyle(inds)), dest, A, inds...)
    return dest
end

function _generate_unsafe_get_layout_index!_body(N::Int)
    quote
        Base.@_inline_meta
        D = eachindex(dest)
        Dy = iterate(D)
        m = refdata(src)
        @inbounds Base.Cartesian.@nloops $N j d -> I[d] begin
            # This condition is never hit, but at the moment
            # the optimizer is not clever enough to split the union without it
            Dy === nothing && return dest
            (idx, state) = Dy
            dest[idx] = m[lyt[NDIndex(Base.Cartesian.@ntuple($N, j))]]
            Dy = iterate(D, state)
        end
        return dest
    end
end

function _generate_unsafe_get_index!_body(N::Int)
    quote
        Base.@_inline_meta
        D = eachindex(dest)
        Dy = iterate(D)
        @inbounds Base.Cartesian.@nloops $N j d -> I[d] begin
            # This condition is never hit, but at the moment
            # the optimizer is not clever enough to split the union without it
            Dy === nothing && return dest
            (idx, state) = Dy
            dest[idx] = unsafe_get_element(src, NDIndex(Base.Cartesian.@ntuple($N, j)))
            Dy = iterate(D, state)
        end
        return dest
    end
end

@generated function unsafe_get_index!(lyt::ArrayIndex, dest, src, I::Vararg{Any,N}) where {N}
    _generate_unsafe_get_layout_index!_body(N)
end
@generated function unsafe_get_index!(::Nothing, dest, src, I::Vararg{Any,N}) where {N}
    _generate_unsafe_get_index!_body(N)
end

_ints2range(x::Integer) = x:x
_ints2range(x::AbstractRange) = x
@inline function unsafe_get_collection(A::CartesianIndices{N}, inds) where {N}
    if (length(inds) === 1 && N > 1) || stride_preserving_index(typeof(inds)) === False()
        return Base._getindex(IndexStyle(A), A, inds...)
    else
        return CartesianIndices(to_axes(A, _ints2range.(inds)))
    end
end
@inline function unsafe_get_collection(A::LinearIndices{N}, inds) where {N}
    if is_linear_indexing(A, inds)
        return @inbounds(eachindex(A)[first(inds)])
    elseif stride_preserving_index(typeof(inds)) === True()
        return LinearIndices(to_axes(A, _ints2range.(inds)))
    else
        return Base._getindex(IndexStyle(A), A, inds...)
    end
end

#################
### setindex! ###
#################
"""
    ArrayInterface.setindex!(A, args...)

Store the given values at the given key or index within a collection.
"""
@propagate_inbounds function setindex!(A, val, args...)
    if can_setindex(A)
        return unsafe_set_index!(A, val, to_indices(A, args))
    else
        error("Instance of type $(typeof(A)) are not mutable and cannot change elements after construction.")
    end
end
@propagate_inbounds function setindex!(A, val; kwargs...)
    return unsafe_set_index!(A, val, to_indices(A, order_named_inds(dimnames(A), kwargs.data)))
end

unsafe_set_index!(A, v, inds::Tuple) = _unsafe_set_index!(_is_element_index(inds), A, v, inds)
_unsafe_set_index!(::False, A, v, inds::Tuple) = unsafe_set_collection!(A, v, inds)
_unsafe_set_index!(::True, A, v, inds::Tuple) = __unsafe_set_index!(A, v, inds)
__unsafe_set_index!(A, v, inds::Tuple{}) = unsafe_set_element!(A, v, ())
function __unsafe_set_index!(A, v, inds::Tuple{Any})
    return unsafe_set_element!(A, v, to_index(A, first(inds)))
end
function __unsafe_set_index!(A, v, inds::Tuple{Any,Vararg{Any}})
    return unsafe_set_element!(A, v, to_index(A, NDIndex(inds)))
end

"""
    unsafe_set_element!(A, val, inds::Tuple)

Sets an element of `A` to `val` at indices `inds`. This method assumes all `inds`
have been checked for being in bounds. Any new array type using `ArrayInterface.setindex!`
must define `unsafe_set_element!(::NewArrayType, val, inds)`.
"""
unsafe_set_element!(a, val, inds) = _unsafe_set_element!(has_parent(a), a, val, inds)
_unsafe_set_element!(::True, a, val, inds) = unsafe_set_element!(parent(a), val, inds)
_unsafe_set_element!(::False, a, val, inds) = @inbounds(parent(a)[inds] = val)

function _unsafe_set_element!(::False, a::AbstractArray2, val, inds)
    unsafe_set_element_error(a, val, inds)
end
unsafe_set_element_error(A, v, i) = throw(MethodError(unsafe_set_element!, (A, v, i)))

function unsafe_set_element!(A::Array{T}, val, ::Tuple{}) where {T}
    Base.arrayset(false, A, convert(T, val)::T, 1)
end
function unsafe_set_element!(A::Array{T}, val, i::Integer) where {T}
    return Base.arrayset(false, A, convert(T, val)::T, Int(i))
end

# This is based on Base._unsafe_setindex!.
"""
    unsafe_set_collection!(A, val, inds)

Sets `inds` of `A` to `val`. `inds` is assumed to have been bounds-checked.
"""
@inline unsafe_set_collection!(A, v, i) = _unsafe_setindex!(A, v, i...)

function _generate_unsafe_setindex!_body(N::Int)
    quote
        x′ = Base.unalias(A, x)
        Base.Cartesian.@nexprs $N d -> (I_d = Base.unalias(A, I[d]))
        idxlens = Base.Cartesian.@ncall $N Base.index_lengths I
        Base.Cartesian.@ncall $N Base.setindex_shape_check x′ (d -> idxlens[d])
        Xy = iterate(x′)
        @inbounds Base.Cartesian.@nloops $N i d->I_d begin
            # This is never reached, but serves as an assumption for
            # the optimizer that it does not need to emit error paths
            Xy === nothing && break
            (val, state) = Xy
            unsafe_set_element!(A, val, NDIndex(Base.Cartesian.@ntuple($N, i)))
            Xy = iterate(x′, state)
        end
        A
    end
end

@generated function _unsafe_setindex!(A, x, I::Vararg{Any,N}) where {N}
    return _generate_unsafe_setindex!_body(N)
end

