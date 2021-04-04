
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
argdims(::ArrayStyle, ::Type{T}) where {N,T<:CartesianIndex{N}} = static(N)
argdims(::ArrayStyle, ::Type{T}) where {N,T<:AbstractArray{CartesianIndex{N}}} = static(N)
argdims(::ArrayStyle, ::Type{T}) where {N,T<:AbstractArray{<:Any,N}} = static(N)
argdims(::ArrayStyle, ::Type{T}) where {N,T<:LogicalIndex{<:Any,<:AbstractArray{Bool,N}}} = static(N)
_argdims(s::ArrayStyle, ::Type{I}, i::StaticInt) where {I} = argdims(s, _get_tuple(I, i))
function argdims(s::ArrayStyle, ::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}}
    return eachop(_argdims, nstatic(Val(N)), s, T)
end

is_element_index(i) = is_element_index(typeof(i))
is_element_index(::Type{T}) where {T} = static(false)
is_element_index(::Type{T}) where {T<:AbstractCartesianIndex} = static(true)
is_element_index(::Type{T}) where {T<:Integer} = static(true)
_is_element_index(::Type{T}, i::StaticInt) where {T} = is_element_index(_get_tuple(T, i))
function is_element_index(::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}}
    return static(all(eachop(_is_element_index, nstatic(Val(N)), T)))
end

"""
    UnsafeIndex(::ArrayStyle, ::Type{I})

`UnsafeIndex` controls how indices that have been bounds checked and converted to
native axes' indices are used to return the stored values of an array. For example,
if the indices at each dimension are single integers then `UnsafeIndex(array, inds)` returns
`UnsafeGetElement()`. Conversely, if any of the indices are vectors then `UnsafeGetCollection()`
is returned, indicating that a new array needs to be reconstructed. This method permits
customizing the terminal behavior of the indexing pipeline based on arguments passed
to `ArrayInterface.getindex`. New subtypes of `UnsafeIndex` should define `promote_rule`.
"""
abstract type UnsafeIndex end

struct UnsafeGetElement <: UnsafeIndex end

struct UnsafeGetCollection <: UnsafeIndex end

UnsafeIndex(x, i) = UnsafeIndex(x, typeof(i))
UnsafeIndex(x, ::Type{I}) where {I} = UnsafeIndex(ArrayStyle(x), I)
UnsafeIndex(s::ArrayStyle, i) = UnsafeIndex(s, typeof(i))
UnsafeIndex(::ArrayStyle, ::Type{I}) where {I} = UnsafeGetElement()
UnsafeIndex(::ArrayStyle, ::Type{I}) where {I<:AbstractArray} = UnsafeGetCollection()

Base.promote_rule(::Type{X}, ::Type{Y}) where {X<:UnsafeIndex,Y<:UnsafeGetElement} = X

@generated function UnsafeIndex(s::ArrayStyle, ::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}}
    if N === 0
        return UnsafeGetElement()
    else
        e = Expr(:call, promote_type)
        for p in T.parameters
            push!(e.args, :(typeof(ArrayInterface.UnsafeIndex(s, $p))))
        end
        return Expr(:block, Expr(:meta, :inline), Expr(:call, e))
    end
end

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
@propagate_inbounds function to_indices(A, args::Tuple)
    if is_linear_indexing(A, args)
        return (to_index(eachindex(IndexLinear(), A), first(args)),)
    else
        return to_indices(A, axes(A), args)
    end
end

@propagate_inbounds to_indices(A, args::Tuple{}) = to_indices(A, axes(A), ())
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{I,Vararg{Any}},) where {I}
    return _to_indices(argdims(A, I), A, axs, args)
end
@propagate_inbounds function _to_indices(::StaticInt{0}, A, axs::Tuple, args::Tuple)
    return (to_index(first(axs), first(args)), to_indices(A, tail(axs), tail(args))...)
end
@propagate_inbounds function _to_indices(::StaticInt{1}, A, axs::Tuple, args::Tuple)
    return (to_index(first(axs), first(args)), to_indices(A, tail(axs), tail(args))...)
end
@propagate_inbounds function _to_indices(::StaticInt{N}, A, axs::Tuple, args::Tuple) where {N}
    axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    return _to_multi_indices(A, axes_front, axes_tail, first(args), tail(args))
end
@propagate_inbounds function _to_multi_indices(
    A,
    axes_front::Tuple,
    axes_tail::Tuple,
    arg::Union{LinearIndices,CartesianIndices},
    args::Tuple
)
    return (
        to_indices(_layout(IndexStyle(A), axes_front), axes(arg))...,
        to_indices(A, axes_tail, args)...,
    )
end
@propagate_inbounds function _to_multi_indices(
    A,
    axes_front::Tuple,
    axes_tail::Tuple,
    arg::AbstractCartesianIndex,
    args::Tuple
)
    return (
        to_indices(_layout(IndexStyle(A), axes_front), Tuple(arg))...,
        to_indices(A, axes_tail, args)...,
    )
end

@propagate_inbounds function _to_multi_indices(A, f::Tuple, l::Tuple, arg, args::Tuple)
    return (to_index(_layout(IndexStyle(A), f), arg), to_indices(A, l, args)...)
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
    @boundscheck checkindex(Bool, x, arg) || throw(BoundsError(x, arg))
    return _int(arg)
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
to_index(::IndexLinear, x, inds::Tuple{Any}) = first(inds)
function to_index(::IndexLinear, x, inds::Tuple{Any,Vararg{Any}})
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

## IndexCartesian ##
to_index(::IndexCartesian, x, arg::Colon) = CartesianIndices(x)
to_index(::IndexCartesian, x, arg::CartesianIndices{0}) = arg
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
to_index(::IndexCartesian, x, i::Integer) = _int2subs(axes(x), i - offset1(x))
@inline function _int2subs(axs::Tuple{Any,Vararg{Any}}, i)
    axis = first(axs)
    len = static_length(axis)
    inext = div(i, len)
    return (_int(i - len * inext + static_first(axis)), _int2subs(tail(axs), inext)...)
end
_int2subs(axs::Tuple{Any}, i) = _int(i + static_first(first(axs)))


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

"""
    ArrayInterface.getindex(A, args...)

Retrieve the value(s) stored at the given key or index within a collection. Creating
another instance of `ArrayInterface.getindex` should only be done by overloading `A`.
Changing indexing based on a given argument from `args` should be done through,
[`to_index`](@ref), or [`to_axis`](@ref).
"""
@propagate_inbounds getindex(A, args...) = unsafe_get_index(A, to_indices(A, args))
@propagate_inbounds getindex(A; kwargs...) = A[order_named_inds(dimnames(A), kwargs.data)...]
@propagate_inbounds getindex(x::Tuple, i::Int) = getfield(x, i)
@propagate_inbounds getindex(x::Tuple, ::StaticInt{i}) where {i} = getfield(x, i)

## unsafe_get_index ##
unsafe_get_index(A, inds::Tuple) = _unsafe_get_index(is_element_index(inds), A, inds)
_unsafe_get_index(::True, A, inds::Tuple) = unsafe_get_element(A, inds)
_unsafe_get_index(::False, A, inds::Tuple) = unsafe_get_collection(A, inds)

"""
    unsafe_get_element(A::AbstractArray{T}, inds::Tuple) -> T

Returns an element of `A` at the indices `inds`. This method assumes all `inds`
have been checked for being in bounds. Any new array type using `ArrayInterface.getindex`
must define `unsafe_get_element(::NewArrayType, inds)`.
"""
unsafe_get_element(a::A, inds) where {A} = _unsafe_get_element(has_parent(A), a, inds)
_unsafe_get_element(::True, a, inds) = unsafe_get_element(parent(a), inds)
_unsafe_get_element(::False, a, inds) = @inbounds(parent(a)[inds...])
_unsafe_get_element(::False, a::AbstractArray2, inds) = unsafe_get_element_error(a, inds)
unsafe_get_element(A::Array, ::Tuple{}) = Base.arrayref(false, A, 1)
unsafe_get_element(A::Array, inds) = Base.arrayref(false, A, Int(to_index(A, inds)))
unsafe_get_element(A::LinearIndices, inds) = Int(to_index(A, inds))
@inline function unsafe_get_element(A::CartesianIndices, inds)
    if length(inds) === 1
        return CartesianIndex(to_index(A, first(inds)))
    else
        return CartesianIndex(Base._to_subscript_indices(A, inds...))
    end
end
unsafe_get_element(A::ReshapedArray, inds) = @inbounds(A[inds...])
unsafe_get_element(A::SubArray, inds) = @inbounds(A[inds...])

unsafe_get_element_error(A, inds) = throw(MethodError(unsafe_get_element, (A, inds)))

# This is based on Base._unsafe_getindex from https://github.com/JuliaLang/julia/blob/c5ede45829bf8eb09f2145bfd6f089459d77b2b1/base/multidimensional.jl#L755.
"""
    unsafe_get_collection(A, inds)

Returns a collection of `A` given `inds`. `inds` is assumed to have been bounds-checked.
"""
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
        Base.@_inline_meta
        D = eachindex(dest)
        Dy = iterate(D)
        @inbounds Base.Cartesian.@nloops $N j d -> I[d] begin
            # This condition is never hit, but at the moment
            # the optimizer is not clever enough to split the union without it
            Dy === nothing && return dest
            (idx, state) = Dy
            dest[idx] = unsafe_get_element(src, Base.Cartesian.@ntuple($N, j))
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
    if is_linear_indexing(A, inds)
        return @inbounds(eachindex(A)[first(inds)])
    elseif stride_preserving_index(typeof(inds)) === True()
        return LinearIndices(to_axes(A, _ints2range.(inds)))
    else
        return Base._getindex(IndexStyle(A), A, inds...)
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
        error("Instance of type $(typeof(A)) are not mutable and cannot change elements after construction.")
    end
end
@propagate_inbounds function setindex!(A, val; kwargs...)
    if has_dimnames(A)
        return setindex!(A, val, order_named_inds(dimnames(A), kwargs.data)...)
    else
        return unsafe_setindex!(A, val, to_indices(A, ()))
    end
end

"""
    unsafe_setindex!(A, val, inds::Tuple)

Sets indices (`inds`) of `A` to `val`. This method assumes that `inds` have already been
bounds-checked. This step of the processing pipeline can be customized by:
"""
unsafe_setindex!(A, val, i::Tuple) = unsafe_setindex!(UnsafeIndex(A, i), A, val, i)
unsafe_setindex!(::UnsafeGetElement, A, val, i::Tuple) = unsafe_set_element!(A, val, i)
unsafe_setindex!(::UnsafeGetCollection, A, v, i::Tuple) = unsafe_set_collection!(A, v, i)

unsafe_set_element_error(A, v, i) = throw(MethodError(unsafe_set_element!, (A, v, i)))

"""
    unsafe_set_element!(A, val, inds::Tuple)

Sets an element of `A` to `val` at indices `inds`. This method assumes all `inds`
have been checked for being in bounds. Any new array type using `ArrayInterface.setindex!`
must define `unsafe_set_element!(::NewArrayType, val, inds)`.
"""
unsafe_set_element!(a, val, inds) = _unsafe_set_element!(has_parent(a), a, val, inds)
_unsafe_set_element!(::True, a, val, inds) = unsafe_set_element!(parent(a), val, inds)
_unsafe_set_element!(::False, a, val,inds) = @inbounds(parent(a)[inds...] = val)
function _unsafe_set_element!(::False, a::AbstractArray2, val, inds)
    unsafe_set_element_error(a, val, inds)
end

function unsafe_set_element!(A::Array{T}, val, inds::Tuple) where {T}
    if length(inds) === 0
        return Base.arrayset(false, A, convert(T, val)::T, 1)
    elseif inds isa Tuple{Vararg{Int}}
        return Base.arrayset(false, A, convert(T, val)::T, inds...)
    else
        throw(MethodError(unsafe_set_element!, (A, inds)))
    end
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
            unsafe_set_element!(A, val, Base.Cartesian.@ntuple($N, i))
            Xy = iterate(x′, state)
        end
        A
    end
end
@generated function _unsafe_setindex!(A, x, I::Vararg{Any,N}) where {N}
    return _generate_unsafe_setindex!_body(N)
end

