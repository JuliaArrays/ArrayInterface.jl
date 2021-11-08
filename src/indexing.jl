
function _to_cartesian(a, i::CanonicalInt)
    @inbounds(CartesianIndices(ntuple(dim -> indices(a, dim), Val(ndims(a))))[i])
end
@inline function _to_linear(a, i::Tuple{CanonicalInt,Vararg{CanonicalInt}})
    _strides2int(offsets(a), size_to_strides(size(a), static(1)), i) + static(1)
end

# code gen
@generated function __to_indices(a::A, inds, ::S, ::NDIndex, ::NDShape) where {A,S,NDIndex,NDShape}
    nd = ndims(A)
    blk = Expr(:block, Expr(:(=), :axs, :(lazy_axes(a))))
    t = Expr(:tuple)
    dim = 0
    ndindex = known(NDIndex)
    ndshape = known(NDShape)
    for i in 1:length(ndindex)
        nidx = ndindex[i]
        nout = ndshape[i]
        if nidx === 1
            dim += 1
            axexpr = _axis_expr(nd, dim)
            if nd < dim && nout === 0
                # drop integers after bounds checking trailing dims
                push!(blk.args, :(to_index($axexpr, @inbounds(getfield(inds, $i)))))
            else
                push!(t.args, :(to_index($axexpr, @inbounds(getfield(inds, $i)))))
            end
        else
            axexpr = Expr(:tuple)
            for j in 1:nidx
                dim += 1
                push!(axexpr.args, _axis_expr(nd, dim))
            end
            ICall = ifelse(S <: IndexLinear, :LinearIndices, :CartesianIndices)
            push!(t.args, :(getfield(to_indices($ICall($axexpr), (@inbounds(getfield(inds, $i)),)), 1)))
        end
    end
    quote
        Compat.@inline
        Base.@_propagate_inbounds_meta
        $blk
        $t
    end
end
function _axis_expr(nd::Int, dim::Int)
    ifelse(nd < dim, static(1):static(1), :(@inbounds(getfield(axs, $dim))))
end

# TODO manage CartesianIndex{0}
# This method just flattens out CartesianIndex, CartesianIndices, and Ellipsis. Although no
# values are ever changed and nothing new is actually created, we still get hit with some
# run time costs if we recurse using lispy approach.
@generated function _splat_indices(::StaticInt{N}, inds::I) where {N,I}
    t = Expr(:tuple)
    out = Expr(:block, Expr(:meta, :inline))
    any_splats = false
    ellipsis_position = 0
    NP = length(I.parameters)
    for i in 1:NP
        Ti = I.parameters[i] 
        if Ti <: Base.AbstractCartesianIndex && !(Ti <: CartesianIndex{0})
            argi = gensym()
            push!(out.args, Expr(:(=), argi, :(Tuple(@inbounds(getfield(inds, $i))))))
            for j in 1:ArrayInterface.known_length(Ti)
                push!(t.args, :(@inbounds(getfield($argi, $j))))
            end
            any_splats = true
        elseif Ti <: CartesianIndices && !(Ti <: CartesianIndices{0})
            argi = gensym()
            push!(out.args, Expr(:(=), argi, :(axes(@inbounds(getfield(inds, $i))))))
            for j in 1:ndims(Ti)
                push!(t.args, :(@inbounds(getfield($argi, $j))))
            end
            any_splats = true
        #=
        elseif Ti <: Ellipsis
            if ellipsis_position == 0
                ellipsis_position = i
            else
                push!(t.args, :(:))
            end
        =#
        else
            push!(t.args, :(@inbounds(getfield(inds, $i))))
        end
    end
    if ellipsis_position != 0
        nremaining = N
        for i in 1:NP
            if i != ellipsis_position
                nremaining -= ndims_index(I.parameters[i])
            end
        end
        for _ in 1:nremaining
            insert!(t.args, ellipsis_position, :(:))
        end
    end
    if any_splats
        push!(out.args, t)
        return out
    else
        return :inds
    end
end


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

"""
    to_indices(A, inds::Tuple) -> Tuple

Maps indexing arguments `inds` to the axes of `A`, ensures they are converted to a native
indexing form, and that they are inbounds. Unless all indices in `inds` return `static(true)`
on a call to [`is_canonical`](@ref), then they each are checked at the axis level with
[`to_index`](@ref).
"""
to_indices(A, ::Tuple{}) = (@boundscheck ndims(A) === 0 || throw(BoundsError(A, ())); ())
# preserve CartesianIndices{0} as they consume a dimension.
to_indices(A, i::Tuple{CartesianIndices{0}}) = i
to_indices(A, ::Tuple{Colon}) = (indices(A),)
to_indices(A, i::Tuple{Slice}) = i
to_indices(A, i::Tuple{Vararg{CanonicalInt}}) = (@boundscheck checkbounds(A, i...); i)
to_indices(A, i::Tuple{AbstractArray{<:Integer}}) = (@boundscheck checkbounds(A, i...); i)
to_indices(A, i::Tuple{LogicalIndex}) = (@boundscheck checkbounds(A, i...); i)
@propagate_inbounds to_indices(A, i::Tuple{LinearIndices}) = to_indices(A, axes(getfield(i,1)))
@propagate_inbounds to_indices(A, i::Tuple{CartesianIndices}) = to_indices(A, axes(getfield(i,1)))
@propagate_inbounds to_indices(A, i::Tuple{AbstractCartesianIndex}) = to_indices(A, Tuple(getfield(i, 1)))
function to_indices(A, i::Tuple{AbstractArray{<:AbstractCartesianIndex{N}}}) where {N}
    @boundscheck checkindex(Bool, ntuple(i->indices(A, i), Val(N)), getfield(i, 1)) || throw(BoundsError(A, i))
    i
end
function to_indices(A, i::Tuple{AbstractArray{Bool,N}}) where {N}
    @boundscheck ntuple(i->indices(A, i), Val(N)) == axes(getfield(i, 1)) || throw(BoundsError(A, i))
    (LogicalIndex(getfield(i, 1)),)
end
# As an optimization, we allow trailing Array{Bool} and BitArray to be linear over trailing dimensions
function to_indices(A::LinearIndices, i::Tuple{Union{Array{Bool}, BitArray}})
    @boundscheck ntuple(i->indices(A, i), Val(N)) == axes(getfield(i, 1)) || throw(BoundsError(A, i))
    (LogicalIndex{Int}(getfield(i, 1)),)
end
_to_indices(A, i::Tuple{Vararg{CanonicalInt}}) = (@boundscheck checkbounds(A, i...); i)
@propagate_inbounds @inline function to_indices(a::A, i::Tuple{Any,Vararg{Any}}) where {A}
    _to_indices(a, _splat_indices(static(ndims(A)), i))
end
@propagate_inbounds @inline function _to_indices(a::A, i::I) where {A,I}
    __to_indices(a, i, IndexStyle(A), ndims_index(I), ndims_shape(I))
end


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
to_index(x, i::Slice) = i
to_index(x, i::Colon) = indices(x)
# TODO If these consume dimensions then do we need to check that that dimension is collapsable?
to_index(x, i::CartesianIndex{0}) = i
function to_index(x, i::AbstractCartesianIndex{1})
    @boundscheck checkindex(Bool, indices(x), i) || throw(BoundsError(x, i))
    @inbounds(i[1])
end
function to_index(x, i::CartesianIndices{1})
    @boundscheck checkindex(Bool, indices(x), getfield(axes(i), 1)) || throw(BoundsError(x, i))
    getfield(axes(i), 1)
end
to_index(x, i::CartesianIndices{0}) = i
function to_index(x, i::AbstractArray{<:Integer})
    @boundscheck checkindex(Bool, indices(x), i) || throw(BoundsError(x, i))
    i
end
function to_index(x, i::AbstractArray{Bool})
    @boundscheck checkindex(Bool, indices(x), i) || throw(BoundsError(x, i))
    LogicalIndex(i)
end
function to_index(x, i::Integer)
    @boundscheck checkindex(Bool, indices(x), i) || throw(BoundsError(x, i))
    canonicalize(i)
end
to_index(x, i::Bool) = (@boundscheck i || throw(BoundsError(x, i)); static_first(x))
@propagate_inbounds to_index(axis, arg) = to_index(IndexStyle(axis), axis, arg)
function to_index(s, axis, arg)
    throw(ArgumentError("invalid index: IndexStyle $s does not support indices of " *
                        "type $(typeof(arg)) for instances of type $(typeof(axis))."))
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

# TODO Need to manage index transformations between nested layers of arrays
function unsafe_getindex(a::A, i::CanonicalInt) where {A}
    if IndexStyle(A) === IndexLinear()
        parent_type(A) <: A && throw(MethodError(unsafe_getindex, (A, i)))
        return unsafe_getindex(parent(a), i)
    else
        return unsafe_getindex(a, _to_cartesian(a, i)...)
    end
end
function unsafe_getindex(a::A, i::CanonicalInt, ii::Vararg{CanonicalInt}) where {A}
    if IndexStyle(A) === IndexLinear()
        return unsafe_getindex(a, _to_linear(a, (i, ii...)))
    else
        parent_type(A) <: A && throw(MethodError(unsafe_getindex, (A, i)))
        return unsafe_getindex(parent(a), i, ii...)
    end
end

unsafe_getindex(a, i::Vararg{Any}) = unsafe_get_collection(a, i)

unsafe_getindex(A::Array) = Base.arrayref(false, A, 1)
unsafe_getindex(A::Array, i::CanonicalInt) = Base.arrayref(false, A, Int(i))

unsafe_getindex(A::LinearIndices, i::CanonicalInt) = Int(i)
unsafe_getindex(A::CartesianIndices, i::CanonicalInt, ii::Vararg{CanonicalInt}) = CartesianIndex(i, ii...)
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
    if map(Base.unsafe_length, axes(dest)) == map(Base.unsafe_length, axs)
        Base._unsafe_getindex!(dest, A, inds...)
    else
        Base.throw_checksize_error(dest, axs)
    end
    return dest
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
# TODO Need to manage index transformations between nested layers of arrays
function unsafe_setindex!(a::A, v, i::CanonicalInt) where {A}
    if IndexStyle(A) === IndexLinear()
        parent_type(A) <: A && throw(MethodError(unsafe_setindex!, (A, v, i)))
        return unsafe_setindex!(parent(a), v, i)
    else
        return unsafe_setindex!(a, v, _to_cartesian(a, i)...)
    end
end
function unsafe_setindex!(a::A, v, i::CanonicalInt, ii::Vararg{CanonicalInt}) where {A}
    if IndexStyle(A) === IndexLinear()
        return unsafe_setindex!(a, v, _to_linear(a, (i, ii...)))
    else
        parent_type(A) <: A && throw(MethodError(unsafe_getindex!, (A, v, i, ii...)))
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

