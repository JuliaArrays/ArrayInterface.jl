
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

Checks if `x` is in  canonical form for indexing. If `x` is already in a canonical form
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

is_linear_indexing(A, args::Tuple{Arg}) where {Arg} = index_ndims(Arg) < 2
is_linear_indexing(A, args::Tuple{Arg,Vararg{Any}}) where {Arg} = false

"""
    to_indices(A, inds::Tuple) -> Tuple

Maps indexing arguments `inds` to the axes of `A`, ensures they are converted to a native
indexing form, and that they are inbounds. Unless all indices in `inds` return `static(true)`
on a call to [`is_canonical`](@ref), then they each are checked at the axis level with
[`to_index`](@ref).
"""
@inline @propagate_inbounds to_indices(A, ::Tuple{}) = to_indices(A, lazy_axes(A), ())
@inline @propagate_inbounds to_indices(A, inds::Tuple) = _to_indices(is_canonical(inds), A, inds)
@inline @propagate_inbounds function to_indices(A, inds::Tuple{LinearIndices})
    to_indices(A, lazy_axes(A), axes(getfield(inds, 1)))
end
@inline @propagate_inbounds function _to_indices(::True, A, inds)
    if isone(sum(index_ndims(inds)))
        @boundscheck if !checkindex(Bool, eachindex(IndexLinear(), A), getfield(inds, 1))
            throw(BoundsError(A, inds))
        end
        return inds
    else
        @boundscheck if !Base.checkbounds_indices(Bool, lazy_axes(A), inds)
            throw(BoundsError(A, inds))
        end
        if ndims(A) < length(inds)
            # FIXME bad solution to trailing indices when canonical
            return permute(inds, nstatic(Val(ndims(A))))
        else
            return inds
        end
    end
end

@inline @propagate_inbounds function _to_indices(::False, A, inds)
    if isone(sum(index_ndims(inds)))
        return (to_index(LazyAxis{:}(A), getfield(inds, 1)),)
    else
        return to_indices(A, lazy_axes(A), inds)
    end
end
@inline @propagate_inbounds function to_indices(A, axs, inds::Tuple{<:AbstractCartesianIndex,Vararg{Any}})
    to_indices(A, axs, (Tuple(getfield(inds, 1))..., tail(inds)...))
end
@inline @propagate_inbounds function to_indices(A, axs, inds::Tuple{I,Vararg{Any}}) where {I}
    _to_indices(index_ndims(I), A, axs, inds)
end

@inline @propagate_inbounds function _to_indices(::StaticInt{1}, A, axs, inds)
    (to_index(_maybe_first(axs), getfield(inds, 1)),
     to_indices(A, _maybe_tail(axs), _maybe_tail(inds))...)
end

@inline @propagate_inbounds function _to_indices(::StaticInt{N}, A, axs, inds) where {N}
    axsfront, axstail = Base.IteratorsMD.split(axs, Val(N))
    if IndexStyle(A) === IndexLinear()
        index = to_index(LinearIndices(axsfront), getfield(inds, 1))
    else
        index = to_index(CartesianIndices(axsfront), getfield(inds, 1))
    end
    return (index, to_indices(A, axstail, _maybe_tail(inds))...)
end
# When used as indices themselves, CartesianIndices can simply become its tuple of ranges
@inline @propagate_inbounds function to_indices(A, axs, inds::Tuple{CartesianIndices, Vararg{Any}})
    to_indices(A, axs, (axes(getfield(inds, 1))..., tail(inds)...))
end
# but preserve CartesianIndices{0} as they consume a dimension.
@inline @propagate_inbounds function to_indices(A, axs, inds::Tuple{CartesianIndices{0},Vararg{Any}})
    (getfield(inds, 1), to_indices(A, _maybe_tail(axs), tail(inds))...)
end
@inline @propagate_inbounds function to_indices(A, axs, ::Tuple{})
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
# Colons get converted to slices by `indices`
@inline to_index(::IndexLinear, axis, arg::Colon) = indices(axis)
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
    @boundscheck Base.checkindex(Bool, axes(x), arg) || throw(BoundsError(x, arg))
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
    @boundscheck Base.checkindex(Bool, axes(x), arg) || throw(BoundsError(x, arg))
    return arg
end
@propagate_inbounds function to_index(::IndexCartesian, x, arg::AbstractArray{<:AbstractCartesianIndex})
    @boundscheck Base.checkindex(Bool, axes(x), arg) || throw(BoundsError(x, arg))
    return arg
end
@propagate_inbounds function to_index(::IndexCartesian, x, arg::AbstractArray{Bool})
    @boundscheck checkbounds(x, arg)
    return LogicalIndex(arg)
end

@propagate_inbounds function to_index(::IndexCartesian, x, arg::Union{Array{Bool}, BitArray})
    @boundscheck checkbounds(x, arg)
    return LogicalIndex{Int}(arg)
end

################
### getindex ###
################
"""
    ArrayInterface.getindex(A, args...)

Retrieve the value(s) stored at the given key or index within a collection. Creating
another instance of `ArrayInterface.getindex` should only be done by overloading `A`.
Changing indexing based on a given argument from `args` should be done through
[`to_index`](@ref).
"""
@propagate_inbounds getindex(A, args...) = unsafe_getindex(A, to_indices(A, args)...)
@propagate_inbounds function getindex(A; kwargs...)
    return unsafe_getindex(A, to_indices(A, order_named_inds(dimnames(A), values(kwargs)))...)
end
@propagate_inbounds getindex(x::Tuple, i::Int) = getfield(x, i)
@propagate_inbounds getindex(x::Tuple, ::StaticInt{i}) where {i} = getfield(x, i)

## unsafe_getindex
function unsafe_getindex(a::A) where {A}
    parent_type(A) <: A && throw(MethodError(unsafe_getindex, (A,)))
    return unsafe_getindex(parent(a))
end
unsafe_getindex(A::Array) = Base.arrayref(false, A, 1)
@inline function unsafe_getindex(A, inds::Vararg{CanonicalInt,N}) where {N}
    buf, lyt = layout(A, static(N))
    @inbounds(buf[lyt[inds...]])
end
@inline function unsafe_getindex(A, inds::Vararg{Any})
    buf, lyt = layout(A, index_dimsum(inds))
    return relayout(getlayout(device(buf), buf, lyt, inds), A, inds)
end

## CartesianIndices/LinearIndices
# TODO replace _ints2range with something that actually indexes each axis
_ints2range(::Tuple{}) = ()
@inline _ints2range(x::Tuple{Any,Vararg{Any}}) = (getfield(x, 1), _ints2range(tail(x))...)
@inline _ints2range(x::Tuple{<:Integer,Vararg{Any}}) = _ints2range(tail(x))

unsafe_getindex(A::CartesianIndices, i::Vararg{CanonicalInt,N}) where {N} = @inbounds(A[CartesianIndex(i)])
@inline function unsafe_getindex(A::CartesianIndices{N}, inds::Vararg{Any}) where {N}
    if (length(inds) === 1 && N > 1) || stride_preserving_index(typeof(inds)) === False()
        return Base._getindex(IndexStyle(A), A, inds...)
    else
        return CartesianIndices(_ints2range(inds))
    end
end

function unsafe_getindex(A::LinearIndices, i::Vararg{CanonicalInt,N}) where {N}
    if N === 1
        return Int(@inbounds(i[1]))
    else
        return Int(@inbounds(_to_linear(A)[NDIndex(i...)]))
    end
end
@inline function unsafe_getindex(A::LinearIndices{N}, inds::Vararg{Any}) where {N}
    if isone(sum(index_ndims(inds)))
        return @inbounds(eachindex(A)[first(inds)])
    elseif stride_preserving_index(typeof(inds)) === True()
        return LinearIndices(_ints2range(inds))
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
    unsafe_setindex!(A, val, to_indices(A, order_named_inds(dimnames(A), values(kwargs)))...)
end

## unsafe_setindex! ##
function unsafe_setindex!(a::A, v) where {A}
    parent_type(A) <: A && throw(MethodError(unsafe_setindex!, (A, v)))
    unsafe_setindex!(parent(a), v)
end
unsafe_setindex!(A::Array{T}, v) where {T} = Base.arrayset(false, A, convert(T, v)::T, 1)
@inline function unsafe_setindex!(A, v, i::Vararg{CanonicalInt,N}) where {N}
    buf, lyt = layout(A, static(N))
    setlayout!(device(buf), buf, lyt, v, i)
end

## layouts - TODO finalize `layout(x, access)` design
@inline layout(x, ::StaticInt{N}) where {N} = _layout(x, buffer(x), ArrayIndex{N}(x))
@inline function _layout(x::X, y::Y, index::ArrayIndex{N}) where {X,Y,N}
    b, i = layout(y, index_dimsum(index))
    return b, compose(i, index)
end
# end recursion b/c no new buffer
_layout(x::X, y::X, i::ArrayIndex) where {X} = x, i
# no new buffer and unkown index transformation, s
_layout(x::X, y::X, ::UnkownIndex{N}) where {X,N} = x, IdentityIndex{N}()
# new buffer, but don't know how to transform indices properly
_layout(x::X, y::Y, ::UnkownIndex{N}) where {X,Y,N} = x, IdentityIndex{N}()

"""
    relayout_constructor(::Type{T}) -> Function

Returns a function that construct a new layout for wrapping sub indices of an array.
This method is called in the context of the indexing arguments and at the array's top level.
Therefore, in the call `relayout_constructor(T)(A, inds) -> layout` the array `A` may be a wrapper
around instance of `T`.

It is assumed that the return of this function can appropriately recompose an layouted array
via `buffer ∘ layout`
"""
relayout_constructor(::Type{T}) where {T} = nothing

@inline function _relayout_constructors(::Type{T}) where {T}
    if parent_type(T) <: T
        return (relayout_constructor(T),)
    else
        return (relayout_constructor(T), _relayout_constructors(parent_type(T))...)
    end
end

"""
    relayout(dest, A, inds)

Derives the function from [`relayout_constructor`](@ref) for each nested parent type of `A`,
which are then used to construct a layout given the arguments `A` and `inds`, and recompose
`dest`. If `relayout_constructor` returns `nothing` then it is not used to in the
recomposing stage.

    A--relayout_constructor(A)--> rc_a--> rc_a(A, inds)--> lyt_a
    |
    parent_type(A) -> B--relayout_constructor(B)--> rc_b--> rc_b(A, inds)--> lyt_b
                      |
                      parent_type(B)--> C --relayout_constructor(C)--> nothing

These results would finally be called as `dest ∘ lyt_b ∘ lyt_a`
"""
relayout(dest, A, inds) = _relayout(_relayout_constructors(typeof(A)), dest, A, inds)
@generated function _relayout(fxns::F, B, A, inds) where {F}
    N = length(F.parameters)
    bexpr = :B
    for i in N:-1:1
        if !(F.parameters[i] <: Nothing)
            bexpr = :(compose($bexpr, getfield(fxns, $i)(A, inds)))
        end
    end
    Expr(:block, Expr(:meta, :inline), bexpr)
end


## CPUTuple
@generated getlayout(::CPUTuple, buf::B, lyt::L, inds::I) where {B,L,I} = _tup_lyt(B, L, I)
function _tup_lyt(B::Type, L::Type, I::Type)
    N = length(I.parameters)
    s = Vector{Int}(undef, N)
    o = Vector{Int}(undef, N)
    static_check = true
    @inbounds for i in 1:N
        s_i = ArrayInterface.known_length(I.parameters[i])
        if s_i === nothing
            static_check = false
            break
        else
            s[i] = s_i
        end
        o_i = ArrayInterface.known_offset1(I.parameters[i])
        if o_i === nothing
            static_check = false
            break
        else
            o[i] = o_i
        end
    end
    if static_check
        t = Expr(:tuple)
        foreach(i->push!(t.args, :(buf[lyt[$(i...)]])), Iterators.product(map((o_i, s_i) -> o_i:(o_i + s_i -1), o, s)...))
        return t
    else # don't know size and offsets so we can't compose tuple statically
        return _idx_lyt(B, L, I)
    end
end

## CPUIndex
function getlayout(::AbstractDevice, buf::B, lyt::L, inds::I) where {B,L,I}
    _idx_lyt(similar(buf, Base.index_shape(inds...)), buf, lyt, inds)
end
@generated function _idx_lyt(dest, src, lyt, inds::I) where {I}
    N = length(I.parameters)
    quote
        Compat.@inline()
        D = eachindex(dest)
        Dy = iterate(D)
        @inbounds Base.Cartesian.@nloops $N j d -> inds[d] begin
            # This condition is never hit, but at the moment
            # the optimizer is not clever enough to split the union without it
            Dy === nothing && return dest
            (idx, state) = Dy
            dest[idx] = src[lyt[NDIndex(Base.Cartesian.@ntuple($N, j))]]
            Dy = iterate(D, state)
        end
        return dest
    end
end

function unsafe_setindex!(A, v, inds::Vararg{Any,N}) where {N}
    buf, lyt = layout(A, index_dimsum(inds))
    return setlayout!(device(buf), buf, lyt, v, inds)
end

function setlayout!(::AbstractDevice, buf::B, lyt::L, v, inds::Tuple{Vararg{CanonicalInt}}) where {B,L}
    @inbounds(Base.setindex!(buf, v, lyt[inds...]))
end
@generated function setlayout!(::AbstractDevice, buf::B, lyt::L, v, inds::Tuple{Vararg{Any,N}}) where {B,L,N}
    _setlayout!(N)
end

function _setlayout!(N::Int)
    quote
        x′ = Base.unalias(buf, v)
        Base.Cartesian.@nexprs $N d -> (I_d = Base.unalias(buf, inds[d]))
        idxlens = Base.Cartesian.@ncall $N Base.index_lengths I
        Base.Cartesian.@ncall $N Base.setindex_shape_check x′ (d -> idxlens[d])
        Xy = iterate(x′)
        @inbounds Base.Cartesian.@nloops $N i d->I_d begin
            # This is never reached, but serves as an assumption for
            # the optimizer that it does not need to emit error paths
            Xy === nothing && break
            (val, state) = Xy
            buf[lyt[NDIndex(Base.Cartesian.@ntuple($N, i))]] = val
            Xy = iterate(x′, state)
        end
    end
end

