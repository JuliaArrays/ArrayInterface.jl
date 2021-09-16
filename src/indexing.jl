
"""
    is_canonical(::Type{I}) -> StaticBool

Returns `True` if instances of `I` can be used for indexing without any further change
(e.g., `Int`, `StaticInt`, `UnitRange{Int}`)
"""
is_canonical(x) = is_canonical(typeof(x))
is_canonical(::Type{T}) where {T} = static(false)
is_canonical(::Type{<:CanonicalInt}) = static(true)
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
function canonical_convert(x::AbstractUnitRange)
    return OptionallyStaticUnitRange(static_first(x), static_last(x))
end

is_linear_indexing(A, args::Tuple{Arg}) where {Arg} = ndims_index(Arg) < 2
is_linear_indexing(A, args::Tuple{Arg,Vararg{Any}}) where {Arg} = false

"""
    to_indices(A, inds::Tuple) -> Tuple

Maps indexing arguments `inds` to the axes of `A`, ensures they are converted to a native
indexing form, and that they are inbounds. Unless all indices in `inds` return `static(true)`
on a call to [`is_canonical`](@ref), then they each are checked at the axis level with
[`to_index`](@ref).
"""
@propagate_inbounds to_indices(A, ::Tuple{}) = to_indices(A, lazy_axes(A), ())
@propagate_inbounds to_indices(A, inds::Tuple) = _to_indices(is_canonical(inds), A, inds)
@propagate_inbounds function to_indices(A, inds::Tuple{LinearIndices})
    to_indices(A, lazy_axes(A), axes(getfield(inds, 1)))
end
@propagate_inbounds function _to_indices(::True, A, inds)
    if isone(sum(ndims_index(inds)))
        @boundscheck if !checkindex(Bool, eachindex(IndexLinear(), A), getfield(inds, 1))
            throw(BoundsError(A, inds))
        end
        return inds
    else
        @boundscheck if !Base.checkbounds_indices(Bool, lazy_axes(A), inds)
            throw(BoundsError(A, inds))
        end
        return inds
    end
end
@propagate_inbounds function _to_indices(::False, A, inds)
    if isone(sum(ndims_index(inds)))
        return (to_index(LazyAxis{:}(A), getfield(inds, 1)),)
    else
        return to_indices(A, lazy_axes(A), inds)
    end
end
@propagate_inbounds function to_indices(A, axs, inds::Tuple{<:AbstractCartesianIndex,Vararg{Any}})
    to_indices(A, axs, (Tuple(getfield(inds, 1))..., tail(inds)...))
end
@propagate_inbounds function to_indices(A, axs, inds::Tuple{I,Vararg{Any}}) where {I}
    _to_indices(ndims_index(I), A, axs, inds)
end

@propagate_inbounds function _to_indices(::StaticInt{1}, A, axs, inds)
    (to_index(_maybe_first(axs), getfield(inds, 1)),
     to_indices(A, _maybe_tail(axs), _maybe_tail(inds))...)
end

@propagate_inbounds function _to_indices(::StaticInt{N}, A, axs, inds) where {N}
    axsfront, axstail = Base.IteratorsMD.split(axs, Val(N))
    if IndexStyle(A) === IndexLinear()
        index = to_index(LinearIndices(axsfront), getfield(inds, 1))
    else
        index = to_index(CartesianIndices(axsfront), getfield(inds, 1))
    end
    return (index, to_indices(A, axstail, _maybe_tail(inds))...)
end
# When used as indices themselves, CartesianIndices can simply become its tuple of ranges
@propagate_inbounds function to_indices(A, axs, inds::Tuple{CartesianIndices, Vararg{Any}})
    to_indices(A, axs, (axes(getfield(inds, 1))..., tail(inds)...))
end
# but preserve CartesianIndices{0} as they consume a dimension.
@propagate_inbounds function to_indices(A, axs, inds::Tuple{CartesianIndices{0},Vararg{Any}})
    (getfield(inds, 1), to_indices(A, _maybe_tail(axs), tail(inds))...)
end
@propagate_inbounds function to_indices(A, axs, ::Tuple{})
    @boundscheck if length(getfield(axs, 1)) != 1
        error("Cannot drop dimension of size $(length(first(axs))).")
    end
    return to_indices(A, _maybe_tail(axs), ())
end
to_indices(A, ::Tuple{}, ::Tuple{}) = ()

# if there aren't anymore axes than we are using trailing dimensions of size one
_maybe_first(::Tuple{}) = static(1):static(1)
_maybe_first(x::Tuple) = getfield(x, 1)
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

# TODO delete this once the layout interface is working
_array_index(::IndexLinear, a, i::CanonicalInt) = i
@inline function _array_index(::IndexStyle, a, i::CanonicalInt)
    CartesianIndices(ntuple(dim -> indices(a, dim), Val(ndims(a))))[i]
end
_array_index(::IndexLinear, a, i::AbstractCartesianIndex{1}) = getfield(Tuple(i), 1)
@inline function _array_index(::IndexLinear, a, i::AbstractCartesianIndex)
    N = ndims(a)
    StrideIndex{N,ntuple(+, Val(N)),nothing}(size_to_strides(size(a), static(1)), offsets(a))[i]
end
_array_index(::IndexStyle, a, i::AbstractCartesianIndex) = i

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
    to_axes(A, inds) -> Tuple

Construct new axes given the corresponding `inds` constructed after
`to_indices(A, args) -> inds`. This method iterates through each pair of axes and
indices calling [`to_axis`](@ref).
"""
@inline function to_axes(A, inds::Tuple)
    if ndims(A) === 1
        return (to_axis(axes(A, 1), first(inds)),)
    elseif isone(sum(ndims_index(inds)))
        return (to_axis(eachindex(IndexLinear(), A), first(inds)),)
    else
        return to_axes(A, axes(A), inds)
    end
end
# drop this dimension
to_axes(A, a::Tuple, i::Tuple{<:Integer,Vararg{Any}}) = to_axes(A, tail(a), tail(i))
to_axes(A, a::Tuple, i::Tuple{I,Vararg{Any}}) where {I} = _to_axes(ndims_index(I), A, a, i)
function _to_axes(::StaticInt{1}, A, axs::Tuple, inds::Tuple)
    return (to_axis(first(axs), first(inds)), to_axes(A, tail(axs), tail(inds))...)
end
@propagate_inbounds function _to_axes(::StaticInt{N}, A, axs::Tuple, inds::Tuple) where {N}
    axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    if IndexStyle(A) === IndexLinear()
        axis = to_axis(LinearIndices(axes_front), getfield(inds, 1))
    else
        axis = to_axis(CartesianIndices(axes_front), getfield(inds, 1))
    end
    return (axis, to_axes(A, axes_tail, tail(inds))...)
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
@propagate_inbounds getindex(A, args...) = unsafe_getindex(A, to_indices(A, args)...)
@propagate_inbounds function getindex(A; kwargs...)
    return unsafe_getindex(A, to_indices(A, order_named_inds(dimnames(A), values(kwargs)))...)
end
@propagate_inbounds getindex(x::Tuple, i::Int) = getfield(x, i)
@propagate_inbounds getindex(x::Tuple, ::StaticInt{i}) where {i} = getfield(x, i)

## unsafe_getindex ##
function unsafe_getindex(a::A) where {A}
    parent_type(A) <: A && throw(MethodError(unsafe_getindex, (A,)))
    return unsafe_getindex(parent(a))
end
function unsafe_getindex(a::A, i::CanonicalInt) where {A}
    idx = _array_index(IndexStyle(A), a, i)
    if idx === i
        parent_type(A) <: A && throw(MethodError(unsafe_getindex, (A, i)))
        return unsafe_getindex(parent(a), i)
    else
        return unsafe_getindex(a, idx)
    end
end
function unsafe_getindex(a::A, i::AbstractCartesianIndex) where {A}
    idx = _array_index(IndexStyle(A), a, i)
    if idx === i
        parent_type(A) <: A && throw(MethodError(unsafe_getindex, (A, i)))
        return unsafe_getindex(parent(a), i)
    else
        return unsafe_getindex(a, idx)
    end
end
function unsafe_getindex(a, i::CanonicalInt, ii::Vararg{CanonicalInt})
    unsafe_getindex(a, NDIndex(i, ii...))
end
unsafe_getindex(a, i::Vararg{Any}) = unsafe_get_collection(a, i)

unsafe_getindex(A::Array) = Base.arrayref(false, A, 1)
unsafe_getindex(A::Array, i::CanonicalInt) = Base.arrayref(false, A, Int(i))

unsafe_getindex(A::LinearIndices, i::CanonicalInt) = Int(i)

unsafe_getindex(A::CartesianIndices, i::AbstractCartesianIndex) = CartesianIndex(i)

unsafe_getindex(A::SubArray, i::CanonicalInt) = @inbounds(A[i])
unsafe_getindex(A::SubArray, i::AbstractCartesianIndex) = @inbounds(A[i])

# This is based on Base._unsafe_getindex from https://github.com/JuliaLang/julia/blob/c5ede45829bf8eb09f2145bfd6f089459d77b2b1/base/multidimensional.jl#L755.
#=
    unsafe_get_collection(A, inds)

Returns a collection of `A` given `inds`. `inds` is assumed to have been bounds-checked.
=#
function unsafe_get_collection(A, inds)
    axs = to_axes(A, inds)
    dest = similar(A, axs)
    if map(Base.unsafe_length, axes(dest)) == map(Base.unsafe_length, axs)
        _unsafe_get_index!(dest, A, inds...) # usually a generated function, don't allow it to impact inference result
    else
        Base.throw_checksize_error(dest, axs)
    end
    return dest
end

function _generate_unsafe_get_index!_body(N::Int)
    quote
        Compat.@inline()
        D = eachindex(dest)
        Dy = iterate(D)
        @inbounds Base.Cartesian.@nloops $N j d -> I[d] begin
            # This condition is never hit, but at the moment
            # the optimizer is not clever enough to split the union without it
            Dy === nothing && return dest
            (idx, state) = Dy
            dest[idx] = unsafe_getindex(src, NDIndex(Base.Cartesian.@ntuple($N, j)))
            Dy = iterate(D, state)
        end
        return dest
    end
end
@generated function _unsafe_get_index!(dest, src, I::Vararg{Any,N}) where {N}
    return _generate_unsafe_get_index!_body(N)
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
    if isone(sum(ndims_index(inds)))
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
        return unsafe_setindex!(A, val, to_indices(A, args)...)
    else
        error("Instance of type $(typeof(A)) are not mutable and cannot change elements after construction.")
    end
end
@propagate_inbounds function setindex!(A, val; kwargs...)
    return unsafe_setindex!(A, val, to_indices(A, order_named_inds(dimnames(A), values(kwargs)))...)
end

## unsafe_setindex! ##
function unsafe_setindex!(a::A, v) where {A}
    parent_type(A) <: A && throw(MethodError(unsafe_setindex!, (A, v)))
    return unsafe_setindex!(parent(a), v)
end
function unsafe_setindex!(a::A, v, i::CanonicalInt) where {A}
    idx = _array_index(IndexStyle(A), a, i)
    if idx === i
        parent_type(A) <: A && throw(MethodError(unsafe_setindex!, (A, v, i)))
        return unsafe_setindex!(parent(a), v, i)
    else
        return unsafe_setindex!(a, v, idx)
    end
end
function unsafe_setindex!(a::A, v, i::AbstractCartesianIndex) where {A}
    idx = _array_index(IndexStyle(A), a, i)
    if idx === i
        parent_type(A) <: A && throw(MethodError(unsafe_setindex!, (A, v, i)))
        return unsafe_setindex!(parent(a), v, i)
    else
        return unsafe_setindex!(a, v, idx)
    end
end
function unsafe_setindex!(a, v, i::CanonicalInt, ii::Vararg{CanonicalInt})
    unsafe_setindex!(a, v, NDIndex(i, ii...))
end
function unsafe_setindex!(A::Array{T}, v) where {T}
    Base.arrayset(false, A, convert(T, v)::T, 1)
end
function unsafe_setindex!(A::Array{T}, v, i::CanonicalInt) where {T}
    return Base.arrayset(false, A, convert(T, v)::T, Int(i))
end

unsafe_setindex!(a, v, i::Vararg{Any}) = unsafe_set_collection!(a, v, i)

# This is based on Base._unsafe_setindex!.
#=
    unsafe_set_collection!(A, val, inds)

Sets `inds` of `A` to `val`. `inds` is assumed to have been bounds-checked.
=#
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
            unsafe_setindex!(A, val, NDIndex(Base.Cartesian.@ntuple($N, i)))
            Xy = iterate(x′, state)
        end
        A
    end
end

@generated function _unsafe_setindex!(A, x, I::Vararg{Any,N}) where {N}
    return _generate_unsafe_setindex!_body(N)
end

