
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
  `A` with each value in `I` ensures that a `CartesianIndex{3}` at the tail of `I` isn't
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
@inline function to_indices(a::A, inds::I) where {A,I}
    _to_indices(a, inds, IndexStyle(A), static(ndims(A)), IndicesInfo(I))
end
@generated function _to_indices(a, inds, ::S, ::StaticInt{N}, ::IndicesInfo{NI,NS,IS}) where {S,N,NI,NS,IS}
    _to_indices_expr(S, N, NI, NS, IS)
end
function _to_indices_expr(S::DataType, N::Int, ni, ns, is)
    blk = Expr(:block, Expr(:meta, :inline))
    # check to see if we are dealing with linear indexing over a multidimensional array
    if length(ni) == 1 && ni[1] === 1
        push!(blk.args, :((to_index(LazyAxis{:}(a), getfield(inds, 1)),)))
    else
        indsexpr = Expr(:tuple)
        ndi = Int[]
        nds = Int[]
        isi = Bool[]
        # 1. unwrap AbstractCartesianIndex, CartesianIndices, Indices
        for i in 1:length(ns)
            ns_i = ns[i]
            if ns_i isa Tuple
                for j in 1:length(ns_i)
                    push!(ndi, 1)
                    push!(nds, ns_i[j])
                    push!(isi, false)
                    push!(indsexpr.args, :(getfield(getfield(getfield(inds, $i), 1), $j)))
                end
            else
                push!(indsexpr.args, :(getfield(inds, $i)))
                push!(ndi, ni[i])
                push!(nds, ns_i)
                push!(isi, is[i])
            end
        end

        # 2. find splat indices
        splat_position = 0
        remaining = N
        for i in eachindex(ndi, nds, isi)
            if isi[i] && splat_position == 0
                splat_position = i
            else
                remaining -= ndi[i]
            end
        end
        if splat_position !== 0
            for _ in 2:remaining
                insert!(ndi, splat_position, 1)
                insert!(nds, splat_position, 1)
                insert!(indsexpr.args, splat_position, indsexpr.args[splat_position])
            end
        end

        # 3. insert `to_index` calls
        dim = 0
        nndi = length(ndi)
        for i in 1:nndi
            ndi_i = ndi[i]
            if ndi_i == 1
                dim += 1
                indsexpr.args[i] = :(to_index($(_axis_expr(N, dim)), $(indsexpr.args[i])))
            else
                subaxs = Expr(:tuple)
                for _ in 1:ndi_i
                    dim += 1
                    push!(subaxs.args, _axis_expr(N, dim))
                end
                if i == nndi && S <: IndexLinear
                    indsexpr.args[i] = :(to_index(LinearIndices($(subaxs)), $(indsexpr.args[i])))
                else
                    indsexpr.args[i] = :(to_index(CartesianIndices($(subaxs)), $(indsexpr.args[i])))
                end
            end
        end
        push!(blk.args, Expr(:(=), :axs, :(lazy_axes(a))))
        push!(blk.args, :(_flatten_tuples($(indsexpr))))
    end
    return blk
end

function _axis_expr(N::Int, d::Int)
    if d <= N
        :(getfield(axs, $d))
    else  # ndims(a)+ can only have indices 1:1
        :($(SOneTo(1)))
    end
end

@generated function _flatten_tuples(inds::I) where {I}
    t = Expr(:tuple)
    for i in 1:known_length(I)
        p = I.parameters[i]
        if p <: Tuple
            for j in 1:known_length(p)
                push!(t.args, :(@inbounds(getfield(getfield(inds, $i), $j))))
            end
        else
            push!(t.args, :(@inbounds(getfield(inds, $i))))
        end
    end
    t
end

"""
    ArrayInterface.to_index([::IndexStyle, ]axis, arg) -> index

Convert the argument `arg` that was originally passed to `ArrayInterface.getindex` for the
dimension corresponding to `axis` into a form for native indexing (`Int`, Vector{Int}, etc.).

`ArrayInterface.to_index` supports passing a function as an index. This function-index is
transformed into a proper index.

```julia
julia> using ArrayInterface, Static

julia> ArrayInterface.to_index(static(1):static(10), 5)
5

julia> ArrayInterface.to_index(static(1):static(10), <(5))
static(1):4

julia> ArrayInterface.to_index(static(1):static(10), <=(5))
static(1):5

julia> ArrayInterface.to_index(static(1):static(10), >(5))
6:static(10)

julia> ArrayInterface.to_index(static(1):static(10), >=(5))
5:static(10)

```

Use of a function-index helps ensure that indices are inbounds

```julia
julia> ArrayInterface.to_index(static(1):static(10), <(12))
static(1):10

julia> ArrayInterface.to_index(static(1):static(10), >(-1))
1:static(10)
```

New axis types with unique behavior should use an `IndexStyle` trait:
```julia
to_index(axis::MyAxisType, arg) = to_index(IndexStyle(axis), axis, arg)
to_index(::MyIndexStyle, axis, arg) = ...
```

"""
to_index(x, i::Slice) = i
to_index(x, ::Colon) = indices(x)
to_index(::LinearIndices{0,Tuple{}}, ::Colon) = Slice(static(1):static(1))
to_index(::CartesianIndices{0,Tuple{}}, ::Colon) = Slice(static(1):static(1))
# logical indexing
to_index(x, i::AbstractArray{Bool}) = LogicalIndex(i)
to_index(x::LinearIndices, i::AbstractArray{Bool}) = LogicalIndex{Int}(i)
# cartesian indexing
@inline to_index(x, i::CartesianIndices{0}) = i
@inline to_index(x, i::CartesianIndices) = getfield(i, :indices)
@inline to_index(x, i::CartesianIndex) = Tuple(i)
@inline to_index(x, i::NDIndex) = Tuple(i)
@inline to_index(x, i::AbstractArray{<:AbstractCartesianIndex}) = i
@inline function to_index(x, i::Base.Fix2{<:Union{typeof(<),typeof(isless)},<:Union{Base.BitInteger,StaticInt}})
    offset1(x):min(_sub1(canonicalize(i.x)), static_lastindex(x))
end
@inline function to_index(x, i::Base.Fix2{typeof(<=),<:Union{Base.BitInteger,StaticInt}})
    offset1(x):min(canonicalize(i.x), static_lastindex(x))
end
@inline function to_index(x, i::Base.Fix2{typeof(>=),<:Union{Base.BitInteger,StaticInt}})
    max(canonicalize(i.x), offset1(x)):static_lastindex(x)
end
@inline function to_index(x, i::Base.Fix2{typeof(>),<:Union{Base.BitInteger,StaticInt}})
    max(_add1(canonicalize(i.x)), offset1(x)):static_lastindex(x)
end
# integer indexing
to_index(x, i::AbstractArray{<:Integer}) = i
to_index(x, @nospecialize(i::StaticInt)) = i
to_index(x, i::Integer) = Int(i)
@inline to_index(x, i) = to_index(IndexStyle(x), x, i)
function to_index(S::IndexStyle, x, i)
    throw(ArgumentError(
        "invalid index: $S does not support indices of type $(typeof(i)) for instances of type $(typeof(x))."
    ))
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
    if ndims(A) === 1 && Base.length(inds) === 1
        return (to_axis(axes(A, 1), first(inds)),)
    elseif Base.length(inds) === 1 && _ndims_shape(inds[1]) === 1
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

"""
    NDIndices{N,M,R,I,T}

A alternative to `CartesianIndices`, whose element type is `NDIndex` instead of `CartesianIndex`.
"""
struct NDIndices{N,M,R,I,T} <: AbstractArray{NDIndex{M,T},N}
    indices::I
    #=
    N: number of dimension of array;
    M: number of dimension of element NDIndex;
    R: number of dimension represented trailing axes who is always 1:1

    if N = M and R = 0, NDIndices is the same as CartesianIndices;
    unlike CartesianIndex, element of indices can be a abstract vector or a CanonicalInt;
    where CanonicalInt donates a hidden dimension, which shows in index but will be skip in getindex;
    =#
    NDIndices{N,M,R}(inds::NTuple{M,Any}) where {N,M,R} =
        new{N,M,R,typeof(inds),NTuple{M,Int}}(inds)
end
function NDIndices(inds::Tuple)
    N = Base.length(inds)
    return NDIndices{N,N,0}(inds)
end
NDIndices(A::AbstractArray) = NDIndices(axes(A))

function unsafe_subindices(parent::NDIndices, inds)
    M = Base.length(eltype(parent))
    shapes = map(_ndims_shape, inds)
    shapes_L, shapes_R = Base.IteratorsMD.split(shapes, Val(M))
    R = _sum_tup(shapes_R)
    N = _sum_tup(shapes_L) + R
    indices = _map_unsafe_getindex(parent.indices, inds)::NTuple{M,Any}
    return NDIndices{N,M,R}(indices)
end

_ndims_shape(x) = _sum_tup(ndims_shape(x))

function _sum_tup(t::Tuple{Vararg{Integer}})
    s = reduce_tup(+, t)
    return isnothing(s) ? 0 : Int(s)
end
_sum_tup(n::Integer) = Int(n)

function _map_unsafe_getindex(x::Tuple, y::Tuple)
    x1 = first(x)
    if ndims_shape(x1) == 0
        return x1, _map_unsafe_getindex(tail(x), y)...
    else # ndims_shape(x1) == 1
        return static_unsafe_getindex(x1, first(y)),
            _map_unsafe_getindex(tail(x), tail(y))...
    end
end
_map_unsafe_getindex(::Tuple{}, ::Tuple) = ()

@inline Base.size(A::NDIndices) = map(length, axes(A))
@inline axes(I::NDIndices{N}) where {N} = _indices_sub(Val(N), I.indices...)
@inline _indices_sub(::Val{N}, ::CanonicalInt, I...) where {N} =
    _indices_sub(Val(N), I...)
@inline _indices_sub(::Val{N} ,i1, I...) where {N} =
    (axes(i1)..., _indices_sub(Val(N-1), I...)...)
@inline _indices_sub(::Val{N}) where {N} = ntuple(_ -> SOneTo(1), Val(N))

ArrayInterfaceCore.ndims_index(::Type{<:NDIndices{N,M}}) where {N,M} = M
ArrayInterfaceCore.ndims_shape(::Type{<:NDIndices{N}}) where {N} = N
function to_index(x, i::NDIndices{N,M,R}) where {N,M,R}
    if N !== 0 && N === M && R === 0 # like CartesianIndices
        inds = getfield(i, :indices)
        if M === 1
            return first(inds)
        else
            return inds
        end
    else
        # TODO: this might be not an efficient way, for better performance
        # we might need to implement unsafe_get_collection for NDIndices
        return i
    end
end
@inline function to_axes(A, a::Tuple, i::Tuple{I,Vararg{Any}}) where {N,I<:NDIndices{N}}
    axes_front, axes_tail = Base.IteratorsMD.split(a, Val(N))
    return (to_axes(A, axes_front, axes(first(i)))..., to_axes(A, axes_tail, tail(i))...)
end

@inline function Base.checkbounds(::Type{Bool}, A::AbstractArray, i::NDIndices)
    Base.checkbounds_indices(Bool, axes(A), (i,))
end
# this is a modification of checkbounds for AbstractArray{CartesianIndex{N}}
# which might should be implemented in Static
@inline function Base.checkbounds_indices(::Type{Bool}, ::Tuple{},
    I::Tuple{NDIndices{N,M},Vararg{Any}}) where {N,M}
    checkindex(Bool, (), I[1]) & Base.checkbounds_indices(Bool, (), tail(I))
end
@inline function Base.checkbounds_indices(::Type{Bool}, IA::Tuple{Any},
    I::Tuple{NDIndices{0},Vararg{Any}})
    Base.checkbounds_indices(Bool, IA, tail(I))
end
@inline function Base.checkbounds_indices(::Type{Bool}, IA::Tuple{Any},
    I::Tuple{NDIndices{N,M},Vararg{Any}}) where {N,M}
    checkindex(Bool, IA, I[1]) & Base.checkbounds_indices(Bool, (), tail(I))
end
@inline function Base.checkbounds_indices(::Type{Bool}, IA::Tuple,
    I::Tuple{NDIndices{N,M},Vararg{Any}}) where {N,M}
    IA1, IArest = Base.IteratorsMD.split(IA, Val(M))
    checkindex(Bool, IA1, I[1]) & Base.checkbounds_indices(Bool, IArest, tail(I))
end
Base.checkindex(::Type{Bool}, inds::Tuple, I::NDIndices) =
    Base.checkbounds_indices(Bool, inds, getfield(I, :indices))

@propagate_inbounds Base.getindex(I::NDIndices{N}, ii::Vararg{CanonicalInt,N}) where {N} =
    (@boundscheck checkbounds(I, ii...); NDIndex(_unsafe_getindex_sub(I.indices, ii)))

function unsafe_getindex(I::NDIndices, i::CanonicalInt)
    if ndims(I) == 1
        return NDIndex(_unsafe_getindex_sub(I.indices, (i,)))
    else
        return unsafe_getindex(I, _to_cartesian(I, i)...)
    end
end
unsafe_getindex(I::NDIndices, i::CanonicalInt, ii::Vararg{CanonicalInt}) =
    NDIndex(_unsafe_getindex_sub(I.indices, (i, ii...)))

function _unsafe_getindex_sub(indices::Tuple, ii::Tuple)
    ind = first(indices)
    if ndims_shape(ind) == 0
        if length(ii) == 0
            return ind, _unsafe_getindex_sub(tail(indices), ())...
        else
            return ind, _unsafe_getindex_sub(tail(indices), ii)...
        end
    else # if ndims_shape(ind) > 0, then length(ii) > 0
        return static_unsafe_getindex(ind, first(ii)),
            _unsafe_getindex_sub(tail(indices), tail(ii))...
    end
end
_unsafe_getindex_sub(::Tuple{}, ::Tuple) = ()
_unsafe_getindex_sub(::Tuple{}, ::Tuple{}) = ()

is_staticrange(ind) = false
is_staticrange(ind::OrdinalRange{Int,Int}) = known_first(ind) !== nothing &&
    known_last(ind) !== nothing && known_step(ind) !== nothing

function static_unsafe_getindex(r, i)
    if is_staticrange(r)
        if is_staticrange(i)
            f = static_first(r) + (static_first(i) - static(1)) * static_step(r)
            s = static_step(r) * static_step(i)
            l = f + s * (static_length(i) - static(1))
            if s === static(1)
                return f:l
            else
                return f:s:l
            end
        elseif is_static(i) === True()
            return static_first(r) + (static_first(i) - static(1)) * static_step(r)
        end
    end
    return @inbounds r[i]
end

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
_output_shape(ind::AbstractRange, inds...) = (length(ind), _output_shape(inds...)...)
_output_shape(::CanonicalInt) = ()
_output_shape(x::AbstractRange) = (length(x),)
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
# check how many axes have length static(1) which can be dropped safely
_n_size_one_axes(sz1, szs...) = (sz1 === static(1)) + _n_size_one_axes(szs...)
_n_size_one_axes() = 0
@inline function unsafe_get_collection(A::NDIndices{N}, inds) where {N}
    if (Base.length(inds) === 1 && N > 1)
        sz = size(A)
        if (N - _n_size_one_axes(sz...)) > 1
            return Base._getindex(IndexStyle(A), A, inds...)
        else
            inds′ = map(x -> x === static(1) ? static(1) : first(inds), sz)
            return unsafe_subindices(A, inds′)
        end
    else
        return unsafe_subindices(A, inds)
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
