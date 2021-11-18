

@inline function _to_cartesian(a, i::CanonicalInt)
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
        if nidx === 0
            # TODO double check this
            # CartesianIndices{0} consumes dimensions
            push!(t.args, :(@inbounds(getfield(inds, $i))))
        elseif nidx === 1
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
            if S <: IndexLinear && i === length(ndindex)
                push!(t.args, :(getfield(to_indices(LinearIndices($axexpr), (@inbounds(getfield(inds, $i)),)), 1)))
            else
                push!(t.args, :(getfield(to_indices(CartesianIndices($axexpr), (@inbounds(getfield(inds, $i)),)), 1)))
            end
        end
    end
    quote
        Compat.@inline
        $blk
        $t
    end
end
function _axis_expr(nd::Int, dim::Int)
    ifelse(nd < dim, static(1):static(1), :(@inbounds(getfield(axs, $dim))))
end

"""
    is_splat_index(::Type{T}) -> StaticBool
    
Returns `static(true)` if `T` is a type that splats across multiple dimensions. 
"""
is_splat_index(x) = is_splat_index(typeof(x))
is_splat_index(::Type) = static(false)
_is_splat_index(::Type{I}, i::StaticInt) where {I} = is_splat_index(_get_tuple(I, i))
function is_splat_index(::Type{I}) where {I<:Tuple}
    eachop(_is_splat_index, nstatic(Val(known_length(I))), I)
end

# TODO manage CartesianIndex{0}
# This method just flattens out CartesianIndex, CartesianIndices, and Ellipsis. Although no
# values are ever changed and nothing new is actually created, we still get hit with some
# run time costs if we recurse using lispy approach.
@generated function _splat_indices(::StaticInt{N}, inds::I, ::IsSplat) where {N,I,IsSplat}
    t = Expr(:tuple)
    out = Expr(:block, Expr(:meta, :inline))
    any_splats = false
    splat_position = 0
    NP = length(I.parameters)
    splat_sym = gensym()
    for i in 1:NP
        Ti = I.parameters[i] 
        if Ti <: Base.AbstractCartesianIndex
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
        elseif splat_position === 0 && known(IsSplat.parameters[i])
            splat_position = i
            push!(out.args, Expr(:(=), splat_sym, :(@inbounds(getfield(inds, $i)))))
            any_splats = true
        else
            push!(t.args, :(@inbounds(getfield(inds, $i))))
        end
    end
    if splat_position !== 0
        # TODO fix this to be generalized to splats
        nremaining = N
        for i in 1:NP
            if i !== splat_position
                nremaining = nremaining - ndims_index(I.parameters[i])
            end
        end
        for _ in 1:nremaining
            insert!(t.args, splat_position, splat_sym)
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
    to_indices(A, I::Tuple) -> Tuple

Convert the tuple I to a tuple that only contains `Int`, `StaticInt`, `AbstractArray{Integer}`,
or `AbstractArray{<:AbstractCartesianIndex}`.

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
* Specializing by dispatch through method definitions like this:
  `to_indices(::ArrayType, ::Tuple{AxisType,Vararg{Any}}, ::Tuple{::IndexType,Vararg{Any}})`
  require an excessive number of hand written methods to avoid ambiguities. Furthermore, if
  `AxisType` is wrapping another axis that should have unique behavior, then unique parametric 
  types need to also be explicitly defined.
* `to_index(axes(A, dim), index)` is called, as opposed to `Base.to_index(A, index)`. The
  `IndexStyle` of the resulting axis is used to allow indirect dispatch on nested axis types
  within `to_index`.
"""
to_indices(A, ::Tuple{}) = (@boundscheck ndims(A) === 0 || throw(BoundsError(A, ())); ())
# preserve CartesianIndices{0} as they consume a dimension.
to_indices(A, i::Tuple{CartesianIndices{0}}) = i
to_indices(A, ::Tuple{Colon}) = (indices(A),)
to_indices(A, i::Tuple{Slice}) = i
to_indices(A, i::Tuple{Vararg{CanonicalInt}}) = i
to_indices(A, i::Tuple{AbstractArray{<:Integer}}) = i
to_indices(A, i::Tuple{LogicalIndex}) = i
to_indices(A, i::Tuple{LinearIndices}) = to_indices(A, axes(getfield(i,1)))
to_indices(A, i::Tuple{CartesianIndices}) = to_indices(A, axes(getfield(i,1)))
@inline function to_indices(A, i::Tuple{AbstractCartesianIndex})
    to_indices(A, Tuple(getfield(i, 1)))
end
to_indices(A, i::Tuple{AbstractArray{<:AbstractCartesianIndex{N}}}) where {N} = i
to_indices(A, i::Tuple{AbstractArray{Bool,N}}) where {N} = (LogicalIndex(getfield(i, 1)),)
# As an optimization, we allow trailing Array{Bool} and BitArray to be linear over trailing dimensions
@inline function to_indices(A::LinearIndices, i::Tuple{Union{Array{Bool}, BitArray}})
    (LogicalIndex{Int}(getfield(i, 1)),)
end
_to_indices(A, i::Tuple{Vararg{CanonicalInt}}) = i
@inline function to_indices(a::A, i::Tuple{Any,Vararg{Any}}) where {A}
    _to_indices(a, _splat_indices(static(ndims(A)), i, is_splat_index(i)))
end
@inline function _to_indices(a::A, i::I) where {A,I}
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
@inline to_index(axis, arg) = to_index(IndexStyle(axis), axis, arg)
to_index(::IndexStyle, x, i::AbstractArray{<:Integer}) = i
to_index(::IndexStyle, x, i::AbstractVector{Bool}) = LogicalIndex{Int}(i)
to_index(::IndexStyle, x, i::StaticInt) = i
to_index(::IndexStyle, x, i::Integer) = Int(i)
@inline function to_index(::IndexStyle, x, i::Bool)
    start = first(x)
    if i
        return start
    else # subtract static value for type stability
        return start - 1
    end
end
@inline function to_index(::IndexStyle, x, i::AbstractCartesianIndex{1})
    to_index(x, @inbounds(i[1]))
end
@inline function to_index(::IndexStyle, x, i::CartesianIndices{1})
    to_index(x, getfield(axes(i), 1))
end
function to_index(s::IndexStyle, axis, arg)
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
@propagate_inbounds function getindex(A, args...)
    inds = to_indices(A, args)
    @boundscheck checkbounds(A, inds...)
    unsafe_getindex(A, inds...)
end
@propagate_inbounds function getindex(A; kwargs...)
    inds = to_indices(A, order_named_inds(dimnames(A), values(kwargs)))
    @boundscheck checkbounds(A, inds...)
    unsafe_getindex(A, inds...)
end
@propagate_inbounds getindex(x::Tuple, i::Int) = getfield(x, i)
@propagate_inbounds getindex(x::Tuple, ::StaticInt{i}) where {i} = getfield(x, i)

## unsafe_getindex ##
function unsafe_getindex(a::A) where {A}
    parent_type(A) <: A && throw(MethodError(unsafe_getindex, (A,)))
    unsafe_getindex(parent(a))
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
@inline function unsafe_getindex(A::Array, i::CanonicalInt, ii::Vararg{CanonicalInt})
    unsafe_getindex(A, _to_linear(A, (i, ii...)))
end

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
    can_setindex(A) || error("Instance of type $(typeof(A)) are not mutable and cannot change elements after construction.")
    inds = to_indices(A, args)
    @boundscheck checkbounds(A, inds...)
    unsafe_setindex!(A, val, inds...)
end
@propagate_inbounds function setindex!(A, val; kwargs...)
    can_setindex(A) || error("Instance of type $(typeof(A)) are not mutable and cannot change elements after construction.")
    inds = to_indices(A, order_named_inds(dimnames(A), values(kwargs)))
    @boundscheck checkbounds(A, inds...)
    unsafe_setindex!(A, val, inds...)
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
