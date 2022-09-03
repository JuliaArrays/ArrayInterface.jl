
function known_lastindex(::Type{T}) where {T}
    if known_offset1(T) === nothing || known_length(T) === nothing
        return nothing
    else
        return known_length(T) - known_offset1(T) + 1
    end
end
known_lastindex(@nospecialize x) = known_lastindex(typeof(x))

@inline static_lastindex(x) = Static.maybe_static(known_lastindex, lastindex, x)

function Base.first(x::AbstractVector, n::StaticInt)
    @boundscheck n < 0 && throw(ArgumentError("Number of elements must be nonnegative"))
    start = offset1(x)
    @inbounds x[start:min((start - one(start)) + n, static_lastindex(x))]
end

function Base.last(x::AbstractVector, n::StaticInt)
    @boundscheck n < 0 && throw(ArgumentError("Number of elements must be nonnegative"))
    stop = static_lastindex(x)
    @inbounds x[max(offset1(x), (stop + one(stop)) - n):stop]
end

"""
    ArrayInterface.to_indices(A, I::Tuple) -> Tuple

Converts the tuple of indexing arguments, `I`, into an appropriate form for indexing into `A`.
Typically, each index should be an `Int`, `StaticInt`, a collection with values of `Int`, or a collection with values of `CartesianIndex`
This is accomplished in three steps after the initial call to `to_indices`:

# Extended help

This implementation differs from that of `Base.to_indices` in the following ways:

*  `to_indices(A, I)` never results in recursive processing of `I` through
  `to_indices(A, axes(A), I)`. This is avoided through the use of an internal `@generated`
  method that aligns calls of `to_indices` and `to_index` based on the return values of
  `ndims_index`. This is beneficial because the compiler currently does not optimize away
  the increased time spent recursing through
    each additional argument that needs converting. For example:
    ```julia
    julia> x = rand(4,4,4,4,4,4,4,4,4,4);

    julia> inds1 = (1, 2, 1, 2, 1, 2, 1, 2, 1, 2);

    julia> inds2 = (1, CartesianIndex(1, 2), 1, CartesianIndex(1, 2), 1, CartesianIndex(1, 2), 1);

    julia> inds3 = (fill(true, 4, 4), 2, fill(true, 4, 4), 2, 1, fill(true, 4, 4), 1);

    julia> @btime Base.to_indices(\$x, \$inds2)
    1.105 μs (12 allocations: 672 bytes)
    (1, 1, 2, 1, 1, 2, 1, 1, 2, 1)

    julia> @btime ArrayInterface.to_indices(\$x, \$inds2)
    0.041 ns (0 allocations: 0 bytes)
    (1, 1, 2, 1, 1, 2, 1, 1, 2, 1)

    julia> @btime Base.to_indices(\$x, \$inds3);
    340.629 ns (14 allocations: 768 bytes)

    julia> @btime ArrayInterface.to_indices(\$x, \$inds3);
    11.614 ns (0 allocations: 0 bytes)

    ```
* Recursing through `to_indices(A, axes, I::Tuple{I1,Vararg{Any}})` is intended to provide
  context for processing `I1`. However, this doesn't tell use how many dimensions are
  consumed by what is in `Vararg{Any}`. Using `ndims_index` to directly align the axes of
  `A` with each value in `I` ensures that a `CartesiaIndex{3}` at the tail of `I` isn't
  incorrectly assumed to only consume one dimension.
* `Base.to_indices` may fail to infer the returned type. This is the case for `inds2` and
  `inds3` in the first bullet on Julia 1.6.4.
* Specializing by dispatch through method definitions like this:
  `to_indices(::ArrayType, ::Tuple{AxisType,Vararg{Any}}, ::Tuple{::IndexType,Vararg{Any}})`
  require an excessive number of hand written methods to avoid ambiguities. Furthermore, if
  `AxisType` is wrapping another axis that should have unique behavior, then unique parametric
  types need to also be explicitly defined.
* `to_index(axes(A, dim), index)` is called, as opposed to `Base.to_index(A, index)`. The
  `IndexStyle` of the resulting axis is used to allow indirect dispatch on nested axis types
  within `to_index`.
"""
to_indices(A, ::Tuple{}) = ()
@inline to_indices(A, inds::Tuple{Vararg{Any}}) = Base.to_indices(A, as_indices(A, inds))
@inline function Base.to_indices(A, inds::Tuple{Vararg{ArrayIndex}})
    Base.to_indices(A, as_indices(A, inds))
end
@inline function Base.to_indices(A, inds::Tuple{Vararg{AxisIndex{<:Any,<:Union{StaticInt,Tuple,Colon}}}})
    mapped_indices = map(Base.Fix1(Base.to_index, A), inds)
    return flatten_tuples(mapped_indices)
end

## to_index
function Base.to_index(A, @nospecialize(i::AxisIndex{<:Union{CartesianIndices{0,Tuple{}},Base.Slice,StaticInt,AbstractArray{<:Integer},AbstractArray{<:AbstractCartesianIndex}}}))
    getfield(i, :index)
end
# FIXME better tracking of trailing dimensions
@inline Base.to_index(A, i::AxisIndex{Colon}) = indices(A, getfield(i, :pdims))
@inline function Base.to_index(A, @nospecialize(i::AxisIndex{<:Union{CartesianIndex,NDIndex,CartesianIndices}}))
    getfield(getfield(i, :index), 1)
end
@inline Base.to_index(A, i::AxisIndex{<:Base.BitInteger}) = Int(getfield(i, :index))
@inline function Base.to_index(A, i::AxisIndex{<:AbstractArray{Bool}})
    if (last(getfield(i, :pdims)) == ndims(A)) && (IndexStyle(A) isa IndexLinear)
        return LogicalIndex{Int}(getfield(i, :index))
    else
        return LogicalIndex(getfield(i, :index))
    end
end
@inline function Base.to_index(A, i::AxisIndex{<:Base.Fix2{<:Union{typeof(<),typeof(isless)},<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, getfield(i, :pdims))
    static_first(x):min(_sub1(canonicalize(getfield(i, :index).x)), static_last(x))
end
@inline function Base.to_index(A, i::AxisIndex{<:Base.Fix2{typeof(<=),<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, getfield(i, :pdims))
    static_first(x):min(canonicalize(getfield(i, :index).x), static_last(x))
end
@inline function Base.to_index(A, i::AxisIndex{<:Base.Fix2{typeof(>=),<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, getfield(i, :pdims))
    max(canonicalize(getfield(i, :index).x), static_first(x)):static_last(x)
end
@inline function Base.to_index(A, i::AxisIndex{<:Base.Fix2{typeof(>),<:Union{Base.BitInteger,StaticInt}}})
    x = lazy_axes(A, getfield(i, :pdims))
    max(_add1(canonicalize(getfield(i, :index).x)), static_first(x)):static_last(x)
end
@inline function Base.to_index(A, i::AxisIndex)
    Base.to_index(CartesianIndices(lazy_axes(A, getfield(i, :pdims))), getfield(i, :index))
end

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
    elseif Base.length(inds) === 1
        return (to_axis(eachindex(IndexLinear(), A), first(inds)),)
    else
        return to_axes(A, axes(A), inds)
    end
end
# drop this dimension
to_axes(A, a::Tuple, i::Tuple{<:CanonicalInt,Vararg{Any}}) = to_axes(A, _maybe_tail(a), tail(i))
to_axes(A, a::Tuple, i::Tuple{I,Vararg{Any}}) where {I} = _to_axes(StaticInt(ndims_index(I)), A, a, i)
function _to_axes(::StaticInt{1}, A, axs::Tuple, inds::Tuple)
    return (to_axis(_maybe_first(axs), first(inds)), to_axes(A, _maybe_tail(axs), tail(inds))...)
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

_maybe_first(::Tuple{}) = static(1):static(1)
_maybe_first(t::Tuple) = first(t)
_maybe_tail(::Tuple{}) = ()
_maybe_tail(t::Tuple) = tail(t)

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
to_axis(S::IndexLinear, axis, inds) = StaticInt(1):length(inds)

"""
    ArrayInterface.getindex(A, args...)

Retrieve the value(s) stored at the given key or index within a collection. Creating
another instance of `ArrayInterface.getindex` should only be done by overloading `A`.
Changing indexing based on a given argument from `args` should be done through,
[`to_index`](@ref), or [`to_axis`](@ref).
"""
function getindex(A, args...)
    inds = to_indices(A, args)
    @boundscheck checkbounds(A, inds...)
    unsafe_getindex(A, inds...)
end
@propagate_inbounds function getindex(A; kwargs...)
    inds = to_indices(A, find_all_dimnames(dimnames(A), static(keys(kwargs)), Tuple(values(kwargs)), :))
    @boundscheck checkbounds(A, inds...)
    unsafe_getindex(A, inds...)
end
@propagate_inbounds getindex(x::Tuple, i::Int) = getfield(x, i)
@propagate_inbounds getindex(x::Tuple, ::StaticInt{i}) where {i} = getfield(x, i)

## unsafe_getindex ##
function unsafe_getindex(a::A) where {A}
    is_forwarding_wrapper(A) || throw(MethodError(unsafe_getindex, (A,)))
    unsafe_getindex(parent(a))
end

# TODO Need to manage index transformations between nested layers of arrays
function unsafe_getindex(a::A, i::CanonicalInt) where {A}
    if IndexStyle(A) === IndexLinear()
        is_forwarding_wrapper(A) || throw(MethodError(unsafe_getindex, (A, i)))
        return unsafe_getindex(parent(a), i)
    else
        return unsafe_getindex(a, _to_cartesian(a, i)...)
    end
end
function unsafe_getindex(a::A, i::CanonicalInt, ii::Vararg{CanonicalInt}) where {A}
    if IndexStyle(A) === IndexLinear()
        return unsafe_getindex(a, _to_linear(a, (i, ii...)))
    else
        is_forwarding_wrapper(A) || throw(MethodError(unsafe_getindex, (A, i)))
        return unsafe_getindex(parent(a), i, ii...)
    end
end

unsafe_getindex(a, i::Vararg{Any}) = unsafe_get_collection(a, i)

unsafe_getindex(A::Array) = Base.arrayref(false, A, 1)
unsafe_getindex(A::Array, i::CanonicalInt) = Base.arrayref(false, A, Int(i))
@inline function unsafe_getindex(A::Array, i::CanonicalInt, ii::Vararg{CanonicalInt})
    unsafe_getindex(A, _to_linear(A, (i, ii...)))
end

unsafe_getindex(A::LinearIndices, i::CanonicalInt) = Int(i)
unsafe_getindex(A::CartesianIndices{N}, ii::Vararg{CanonicalInt,N}) where {N} = CartesianIndex(ii...)
unsafe_getindex(A::CartesianIndices, ii::Vararg{CanonicalInt}) =
    unsafe_getindex(A, Base.front(ii)...)
unsafe_getindex(A::CartesianIndices, i::CanonicalInt) = @inbounds(A[i])

unsafe_getindex(A::ReshapedArray, i::CanonicalInt) = @inbounds(parent(A)[i])
function unsafe_getindex(A::ReshapedArray, i::CanonicalInt, ii::Vararg{CanonicalInt})
    @inbounds(parent(A)[_to_linear(A, (i, ii...))])
end

unsafe_getindex(A::SubArray, i::CanonicalInt) = @inbounds(A[i])
unsafe_getindex(A::SubArray, i::CanonicalInt, ii::Vararg{CanonicalInt}) = @inbounds(A[i, ii...])

# This is based on Base._unsafe_getindex from https://github.com/JuliaLang/julia/blob/c5ede45829bf8eb09f2145bfd6f089459d77b2b1/base/multidimensional.jl#L755.
#=
    unsafe_get_collection(A, inds)

Returns a collection of `A` given `inds`. `inds` is assumed to have been bounds-checked.
=#
function unsafe_get_collection(A, inds)
    axs = to_axes(A, inds)
    dest = similar(A, axs)
    if map(length, axes(dest)) == map(length, axs)
        Base._unsafe_getindex!(dest, A, inds...)
    else
        Base.throw_checksize_error(dest, axs)
    end
    return dest
end
_ints2range(x::CanonicalInt) = x:x
_ints2range(x::AbstractRange) = x
# apply _ints2range to front N elements
_ints2range_front(::Val{N}, ind, inds...) where {N} =
    (_ints2range(ind), _ints2range_front(Val(N - 1), inds...)...)
_ints2range_front(::Val{0}, ind, inds...) = ()
_ints2range_front(::Val{0}) = ()
# get output shape with given indices
_output_shape(::CanonicalInt, inds...) = _output_shape(inds...)
_output_shape(ind::AbstractRange, inds...) = (Base.length(ind), _output_shape(inds...)...)
_output_shape(::CanonicalInt) = ()
_output_shape(x::AbstractRange) = (Base.length(x),)
@inline function unsafe_get_collection(A::CartesianIndices{N}, inds) where {N}
    if (Base.length(inds) === 1 && N > 1) || stride_preserving_index(typeof(inds)) === False()
        return Base._getindex(IndexStyle(A), A, inds...)
    else
        return reshape(
            CartesianIndices(_ints2range_front(Val(N), inds...)),
            _output_shape(inds...)
        )
    end
end
_known_first_isone(ind) = known_first(ind) !== nothing && isone(known_first(ind))
@inline function unsafe_get_collection(A::LinearIndices{N}, inds) where {N}
    if Base.length(inds) === 1 && ndims_index(typeof(first(inds))) === 1
        return @inbounds(eachindex(A)[first(inds)])
    elseif stride_preserving_index(typeof(inds)) === True() &&
            reduce_tup(&, map(_known_first_isone, inds))
        # create a LinearIndices when first(ind) != 1 is imposable
        return reshape(
            LinearIndices(_ints2range_front(Val(N), inds...)),
            _output_shape(inds...)
        )
    else
        return Base._getindex(IndexStyle(A), A, inds...)
    end
end

"""
    ArrayInterface.setindex!(A, args...)

Store the given values at the given key or index within a collection.
"""
@propagate_inbounds function setindex!(A, val, args...)
    can_setindex(A) || error("Instance of type $(typeof(A)) are not mutable and cannot change elements after construction.")
    inds = to_indices(A, args)
    @boundscheck checkbounds(A, inds...)
    unsafe_setindex!(A, val, inds...)
end
@propagate_inbounds function setindex!(A, val; kwargs...)
    can_setindex(A) || error("Instance of type $(typeof(A)) are not mutable and cannot change elements after construction.")
    inds = to_indices(A, find_all_dimnames(dimnames(A), static(keys(kwargs)), Tuple(values(kwargs)), :))
    @boundscheck checkbounds(A, inds...)
    unsafe_setindex!(A, val, inds...)
end

## unsafe_setindex! ##
function unsafe_setindex!(a::A, v) where {A}
    is_forwarding_wrapper(A) || throw(MethodError(unsafe_setindex!, (A, v)))
    return unsafe_setindex!(parent(a), v)
end
# TODO Need to manage index transformations between nested layers of arrays
function unsafe_setindex!(a::A, v, i::CanonicalInt) where {A}
    if IndexStyle(A) === IndexLinear()
        is_forwarding_wrapper(A) || throw(MethodError(unsafe_setindex!, (A, v, i)))
        return unsafe_setindex!(parent(a), v, i)
    else
        return unsafe_setindex!(a, v, _to_cartesian(a, i)...)
    end
end
function unsafe_setindex!(a::A, v, i::CanonicalInt, ii::Vararg{CanonicalInt}) where {A}
    if IndexStyle(A) === IndexLinear()
        return unsafe_setindex!(a, v, _to_linear(a, (i, ii...)))
    else
        is_forwarding_wrapper(A) || throw(MethodError(unsafe_setindex!, (A, v, i, ii...)))
        return unsafe_setindex!(parent(a), v, i, ii...)
    end
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
unsafe_set_collection!(A, v, i) = Base._unsafe_setindex!(IndexStyle(A), A, v, i...)
