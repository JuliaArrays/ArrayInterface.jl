
_cartesian_index(i::Tuple{Vararg{Int}}) = CartesianIndex(i)
_cartesian_index(::Any) = missing

"""
    known_first(::Type{T}) -> Union{Int,Missing}

If `first` of an instance of type `T` is known at compile time, return it.
Otherwise, return `missing`.

```julia
julia> ArrayInterface.known_first(typeof(1:4))
missing

julia> ArrayInterface.known_first(typeof(Base.OneTo(4)))
1
```
"""
known_first(x) = known_first(typeof(x))
function known_first(::Type{T}) where {T}
    if parent_type(T) <: T
        return missing
    else
        return known_first(parent_type(T))
    end
end
known_first(::Type{Base.OneTo{T}}) where {T} = 1
function known_first(::Type{T}) where {N,R,T<:CartesianIndices{N,R}}
    _cartesian_index(ntuple(i -> known_first(R.parameters[i]), Val(N)))
end

"""
    known_last(::Type{T}) -> Union{Int,Missing}

If `last` of an instance of type `T` is known at compile time, return it.
Otherwise, return `missing`.

```julia
julia> ArrayInterface.known_last(typeof(1:4))
missing

julia> ArrayInterface.known_first(typeof(static(1):static(4)))
4

```
"""
known_last(x) = known_last(typeof(x))
function known_last(::Type{T}) where {T}
    if parent_type(T) <: T
        return missing
    else
        return known_last(parent_type(T))
    end
end
function known_last(::Type{T}) where {N,R,T<:CartesianIndices{N,R}}
    _cartesian_index(ntuple(i -> known_last(R.parameters[i]), Val(N)))
end

"""
    known_step(::Type{T}) -> Union{Int,Missing}

If `step` of an instance of type `T` is known at compile time, return it.
Otherwise, return `missing`.

```julia
julia> ArrayInterface.known_step(typeof(1:2:8))
missing

julia> ArrayInterface.known_step(typeof(1:4))
1

```
"""
known_step(x) = known_step(typeof(x))
function known_step(::Type{T}) where {T}
    if parent_type(T) <: T
        return missing
    else
        return known_step(parent_type(T))
    end
end
known_step(::Type{<:AbstractUnitRange}) = 1

"""
    OptionallyStaticUnitRange(start, stop) <: AbstractUnitRange{Int}

Similar to `UnitRange` except each field may be an `Int` or `StaticInt`. An
`OptionallyStaticUnitRange` is intended to be constructed internally from other valid
indices. Therefore, users should not expect the same checks are used to ensure construction
of a valid `OptionallyStaticUnitRange` as a `UnitRange`.
"""
struct OptionallyStaticUnitRange{F<:CanonicalInt,L<:CanonicalInt} <: AbstractUnitRange{Int}
    start::F
    stop::L

    function OptionallyStaticUnitRange(start::CanonicalInt, stop::CanonicalInt)
        new{typeof(start),typeof(stop)}(start, stop)
    end
   function OptionallyStaticUnitRange(start::Integer, stop::Integer)
        OptionallyStaticUnitRange(canonicalize(start), canonicalize(stop))
    end
    function OptionallyStaticUnitRange(x::AbstractRange)
        step(x) == 1 && return OptionallyStaticUnitRange(static_first(x), static_last(x))

        errmsg(x) = throw(ArgumentError("step must be 1, got $(step(x))")) # avoid GC frame
        errmsg(x)
    end
    OptionallyStaticUnitRange{F,L}(x::AbstractRange) where {F,L} = OptionallyStaticUnitRange(x)
    function OptionallyStaticUnitRange{StaticInt{F},StaticInt{L}}() where {F,L}
        new{StaticInt{F},StaticInt{L}}()
    end
end

"""
    OptionallyStaticStepRange(start, step, stop) <: OrdinalRange{Int,Int}

Similarly to [`OptionallyStaticUnitRange`](@ref), `OptionallyStaticStepRange` permits
a combination of static and standard primitive `Int`s to construct a range. It
specifically enables the use of ranges without a step size of 1. It may be constructed
through the use of `OptionallyStaticStepRange` directly or using static integers with
the range operator (i.e., `:`).

```julia
julia> using ArrayInterface

julia> x = ArrayInterface.static(2);

julia> x:x:10
static(2):static(2):10

julia> ArrayInterface.OptionallyStaticStepRange(x, x, 10)
static(2):static(2):10

```
"""
struct OptionallyStaticStepRange{F<:CanonicalInt,S<:CanonicalInt,L<:CanonicalInt} <: OrdinalRange{Int,Int}
    start::F
    step::S
    stop::L

    function OptionallyStaticStepRange(start::CanonicalInt, step::CanonicalInt, stop::CanonicalInt)
        lst = _steprange_last(start, step, stop)
        new{typeof(start),typeof(step),typeof(lst)}(start, step, lst)
    end
    function OptionallyStaticStepRange(start::Integer, step::Integer, stop::Integer)
        OptionallyStaticStepRange(canonicalize(start), canonicalize(step), canonicalize(stop))
    end
    function OptionallyStaticStepRange(x::AbstractRange)
        return OptionallyStaticStepRange(static_first(x), static_step(x), static_last(x))
    end
end

# to make StepRange constructor inlineable, so optimizer can see `step` value
@inline function _steprange_last(start::StaticInt, step::StaticInt, stop::StaticInt)
    return StaticInt(_steprange_last(Int(start), Int(step), Int(stop)))
end
@inline function _steprange_last(start::Integer, step::StaticInt, stop::StaticInt)
    if step === one(step)
        # we don't need to check the `stop` if we know it acts like a unit range
        return stop
    else
        return _steprange_last(start, Int(step), Int(stop))
    end
end
@inline function _steprange_last(start::Integer, step::Integer, stop::Integer)
    z = zero(step)
    if step === z
        throw(ArgumentError("step cannot be zero"))
    else
        if stop == start
            return Int(stop)
        else
            if step > z
                if stop > start
                    return stop - Int(unsigned(stop - start) % step)
                else
                    return Int(start - one(start))
                end
            else
                if stop > start
                    return Int(start + one(start))
                else
                    return stop + Int(unsigned(start - stop) % -step)
                end
            end
        end
    end
end

"""
    SUnitRange(start::Int, stop::Int)

An alias for `OptionallyStaticUnitRange` where both the start and stop are known statically.
"""
const SUnitRange{F,L} = OptionallyStaticUnitRange{StaticInt{F},StaticInt{L}}
SUnitRange(start::Int, stop::Int) = SUnitRange{start,stop}()

"""
    SOneTo(n::Int)

An alias for `OptionallyStaticUnitRange` usfeul for statically sized axes.
"""
const SOneTo{L} = SUnitRange{1,L}
SOneTo(n::Int) = SOneTo{n}()

const OptionallyStaticRange = Union{<:OptionallyStaticUnitRange,<:OptionallyStaticStepRange}


known_first(::Type{<:OptionallyStaticUnitRange{StaticInt{F}}}) where {F} = F::Int
known_first(::Type{<:OptionallyStaticStepRange{StaticInt{F}}}) where {F} = F::Int

known_step(::Type{<:OptionallyStaticStepRange{<:Any,StaticInt{S}}}) where {S} = S::Int

known_last(::Type{<:OptionallyStaticUnitRange{<:Any,StaticInt{L}}}) where {L} = L::Int
known_last(::Type{<:OptionallyStaticStepRange{<:Any,<:Any,StaticInt{L}}}) where {L} = L::Int

@inline function Base.first(r::OptionallyStaticRange)::Int
    if known_first(r) === missing
        return getfield(r, :start)
    else
        return known_first(r)
    end
end
function Base.step(r::OptionallyStaticStepRange)::Int
    if known_step(r) === missing
        return getfield(r, :step)
    else
        return known_step(r)
    end
end
@inline function Base.last(r::OptionallyStaticRange)::Int
    if known_last(r) === missing
        return getfield(r, :stop)
    else
        return known_last(r)
    end
end

Base.:(:)(L::Integer, ::StaticInt{U}) where {U} = OptionallyStaticUnitRange(L, StaticInt(U))
Base.:(:)(::StaticInt{L}, U::Integer) where {L} = OptionallyStaticUnitRange(StaticInt(L), U)
function Base.:(:)(::StaticInt{L}, ::StaticInt{U}) where {L,U}
    return OptionallyStaticUnitRange(StaticInt(L), StaticInt(U))
end
function Base.:(:)(::StaticInt{F}, ::StaticInt{S}, ::StaticInt{L}) where {F,S,L}
    return OptionallyStaticStepRange(StaticInt(F), StaticInt(S), StaticInt(L))
end
function Base.:(:)(start::Integer, ::StaticInt{S}, ::StaticInt{L}) where {S,L}
    return OptionallyStaticStepRange(start, StaticInt(S), StaticInt(L))
end
function Base.:(:)(::StaticInt{F}, ::StaticInt{S}, stop::Integer) where {F,S}
    return OptionallyStaticStepRange(StaticInt(F), StaticInt(S), stop)
end
function Base.:(:)(::StaticInt{F}, step::Integer, ::StaticInt{L}) where {F,L}
    return OptionallyStaticStepRange(StaticInt(F), step, StaticInt(L))
end
function Base.:(:)(start::Integer, step::Integer, ::StaticInt{L}) where {L}
    return OptionallyStaticStepRange(start, step, StaticInt(L))
end
function Base.:(:)(start::Integer, ::StaticInt{S}, stop::Integer) where {S}
    return OptionallyStaticStepRange(start, StaticInt(S), stop)
end
function Base.:(:)(::StaticInt{F}, step::Integer, stop::Integer) where {F}
    return OptionallyStaticStepRange(StaticInt(F), step, stop)
end
Base.:(:)(start::StaticInt{F}, ::StaticInt{1}, stop::StaticInt{L}) where {F,L} = start:stop
Base.:(:)(start::Integer, ::StaticInt{1}, stop::StaticInt{L}) where {L} = start:stop
Base.:(:)(start::StaticInt{F}, ::StaticInt{1}, stop::Integer) where {F} = start:stop
function Base.:(:)(start::Integer, ::StaticInt{1}, stop::Integer)
    OptionallyStaticUnitRange(start, stop)
end

Base.isempty(r::OptionallyStaticUnitRange{One}) = last(r) <= 0
Base.isempty(r::OptionallyStaticUnitRange) = first(r) > last(r)
function Base.isempty(r::OptionallyStaticStepRange)
    (r.start != r.stop) & ((r.step > 0) != (r.stop > r.start))
end

function Base.checkindex(
    ::Type{Bool},
    ::SUnitRange{F1,L1},
    ::SUnitRange{F2,L2}
) where {F1,L1,F2,L2}

    (F1::Int <= F2::Int) && (L1::Int >= L2::Int)
end

@propagate_inbounds function Base.getindex(
    r::OptionallyStaticUnitRange,
    s::AbstractUnitRange{<:Integer},
)
    @boundscheck checkbounds(r, s)
    f = static_first(r)
    fnew = f - one(f)
    return (fnew+static_first(s)):(fnew+static_last(s))
end

@propagate_inbounds function Base.getindex(x::OptionallyStaticUnitRange{StaticInt{1}}, i::Int)
    @boundscheck checkbounds(x, i)
    i
end
@propagate_inbounds function Base.getindex(x::OptionallyStaticUnitRange, i::Int)
    val = first(x) + (i - 1)
    @boundscheck ((i < 1) || val > last(x)) && throw(BoundsError(x, i))
    val::Int
end

@inline _try_static(::StaticInt{N}, ::StaticInt{N}) where {N} = StaticInt{N}()
@inline function _try_static(::StaticInt{M}, ::StaticInt{N}) where {M,N}
    @assert false "Unequal Indices: StaticInt{$M}() != StaticInt{$N}()"
end
@noinline unequal_error(x,y) = @assert false "Unequal Indices: x == $x != $y == y"
@inline function check_equal(x, y)
    x == y || unequal_error(x,y)
end
@propagate_inbounds function _try_static(::StaticInt{N}, x) where {N}
    @boundscheck check_equal(StaticInt{N}(), x)
    return StaticInt{N}()
end
@propagate_inbounds function _try_static(x, ::StaticInt{N}) where {N}
    @boundscheck check_equal(x, StaticInt{N}())
    return StaticInt{N}()
end
@propagate_inbounds function _try_static(x, y)
    @boundscheck check_equal(x, y)
    return x
end

## length
@inline function known_length(::Type{T}) where {T<:OptionallyStaticUnitRange}
    return _range_length(known_first(T), known_last(T))
end

@inline function known_length(::Type{T}) where {T<:OptionallyStaticStepRange}
    _range_length(known_first(T), known_step(T), known_last(T))
end

Base.lastindex(x::OptionallyStaticRange) = length(x)
Base.length(r::OptionallyStaticUnitRange) = _range_length(static_first(r), static_last(r))
@inline function Base.length(r::OptionallyStaticStepRange)
    if isempty(r)
        return 0
    else
        return _range_length(static_first(r), static_step(r), static_last(r))
    end
end
_range_length(start, stop) = missing
function _range_length(start::CanonicalInt, stop::CanonicalInt)
    if start > stop
        return 0
    else
        return Int((stop - start) + 1)
    end
end
_range_length(start::CanonicalInt, ::One, stop::CanonicalInt) = _range_length(start, stop)
@inline function _range_length(start::CanonicalInt, step::CanonicalInt, stop::CanonicalInt)
    if step > 1
        return Base.checked_add(Int(div(unsigned(stop - start), step)), 1)
    elseif step < -1
        return Base.checked_add(Int(div(unsigned(start - stop), -step)), 1)
    elseif step > 0
        return Base.checked_add(Int(div(Base.checked_sub(stop, start), step)), 1)
    else
        return Base.checked_add(Int(div(Base.checked_sub(start, stop), -step)), 1)
    end
end
_range_length(start, step, stop) = missing

Base.AbstractUnitRange{Int}(r::OptionallyStaticUnitRange) = r
function Base.AbstractUnitRange{T}(r::OptionallyStaticUnitRange) where {T}
    if known_first(r) === 1 && T <: Integer
        return OneTo{T}(last(r))
    else
        return UnitRange{T}(first(r), last(r))
    end
end

Base.eachindex(r::OptionallyStaticRange) = One():static_length(r)
@inline function Base.iterate(r::OptionallyStaticRange)
    isempty(r) && return missing
    fi = Int(first(r));
    fi, fi
end
function Base.iterate(::SUnitRange{F,L}) where {F,L}
    if L::Int < F::Int
        return missing
    else
        return (F::Int, F::Int)
    end
end
function Base.iterate(::SOneTo{n}, s::Int) where {n}
    if s < n::Int
        s2 = s + 1
        return (s2, s2)
    else
        return missing
    end
end

Base.to_shape(x::OptionallyStaticRange) = length(x)
Base.to_shape(x::Slice{T}) where {T<:OptionallyStaticRange} = length(x)
Base.axes(S::Slice{<:OptionallyStaticUnitRange{One}}) = (S.indices,)
Base.axes(S::Slice{<:OptionallyStaticRange}) = (Base.IdentityUnitRange(S.indices),)

Base.axes(x::OptionallyStaticRange) = (Base.axes1(x),)
Base.axes1(x::OptionallyStaticRange) = eachindex(x)
Base.axes1(x::Slice{<:OptionallyStaticUnitRange{One}}) = x.indices
Base.axes1(x::Slice{<:OptionallyStaticRange}) = Base.IdentityUnitRange(x.indices)

Base.:(-)(r::OptionallyStaticRange) = -static_first(r):-static_step(r):-static_last(r)

Base.reverse(r::OptionallyStaticUnitRange) = static_last(r):static(-1):static_first(r)
function Base.reverse(r::OptionallyStaticStepRange)
    OptionallyStaticStepRange(static_last(r), -static_step(r), static_first(r))
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(r::OptionallyStaticUnitRange))
    print(io, "$(getfield(r, :start)):$(getfield(r, :stop))")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize(r::OptionallyStaticStepRange))
    print(io, "$(getfield(r, :start)):$(getfield(r, :step)):$(getfield(r, :stop))")
end

@inline function Base.getproperty(x::OptionallyStaticRange, s::Symbol)
    if s === :start
        return first(x)
    elseif s === :step
        return step(x)
    elseif s === :stop
        return last(x)
    else
        error("$x has no property $s")
    end
end

"""
  reduce_tup(f::F, inds::Tuple{Vararg{Any,N}}) where {F,N}

An optimized `reduce` for tuples. `Base.reduce`'s `afoldl` will often not inline.
Additionally, `reduce_tup` attempts to order the reduction in an optimal manner.

```julia
julia> using StaticArrays, ArrayInterface, BenchmarkTools

julia> rsum(v::SVector) = ArrayInterface.reduce_tup(+, v.data)
rsum (generic function with 2 methods)

julia> for n ∈ 2:16
           @show n
           v = @SVector rand(n)
           s1 = @btime  sum(\$(Ref(v))[])
           s2 = @btime rsum(\$(Ref(v))[])
       end
n = 2
  0.863 ns (0 allocations: 0 bytes)
  0.863 ns (0 allocations: 0 bytes)
n = 3
  0.862 ns (0 allocations: 0 bytes)
  0.863 ns (0 allocations: 0 bytes)
n = 4
  0.862 ns (0 allocations: 0 bytes)
  0.862 ns (0 allocations: 0 bytes)
n = 5
  1.074 ns (0 allocations: 0 bytes)
  0.864 ns (0 allocations: 0 bytes)
n = 6
  0.864 ns (0 allocations: 0 bytes)
  0.862 ns (0 allocations: 0 bytes)
n = 7
  1.075 ns (0 allocations: 0 bytes)
  0.864 ns (0 allocations: 0 bytes)
n = 8
  1.077 ns (0 allocations: 0 bytes)
  0.865 ns (0 allocations: 0 bytes)
n = 9
  1.081 ns (0 allocations: 0 bytes)
  0.865 ns (0 allocations: 0 bytes)
n = 10
  1.195 ns (0 allocations: 0 bytes)
  0.867 ns (0 allocations: 0 bytes)
n = 11
  1.357 ns (0 allocations: 0 bytes)
  1.400 ns (0 allocations: 0 bytes)
n = 12
  1.543 ns (0 allocations: 0 bytes)
  1.074 ns (0 allocations: 0 bytes)
n = 13
  1.702 ns (0 allocations: 0 bytes)
  1.077 ns (0 allocations: 0 bytes)
n = 14
  1.913 ns (0 allocations: 0 bytes)
  0.867 ns (0 allocations: 0 bytes)
n = 15
  2.076 ns (0 allocations: 0 bytes)
  1.077 ns (0 allocations: 0 bytes)
n = 16
  2.273 ns (0 allocations: 0 bytes)
  1.078 ns (0 allocations: 0 bytes)
```

More importantly, `reduce_tup(_pick_range, inds)` often performs better than `reduce(_pick_range, inds)`.
```julia
julia> using ArrayInterface, BenchmarkTools, Static

julia> inds = (Base.OneTo(100), 1:100, 1:static(100))
(Base.OneTo(100), 1:100, 1:static(100))

julia> @btime reduce(ArrayInterface._pick_range, \$(Ref(inds))[])
  6.405 ns (0 allocations: 0 bytes)
Base.Slice(static(1):static(100))

julia> @btime ArrayInterface.reduce_tup(ArrayInterface._pick_range, \$(Ref(inds))[])
  2.570 ns (0 allocations: 0 bytes)
Base.Slice(static(1):static(100))

julia> inds = (Base.OneTo(100), 1:100, 1:UInt(100))
(Base.OneTo(100), 1:100, 0x0000000000000001:0x0000000000000064)

julia> @btime reduce(ArrayInterface._pick_range, \$(Ref(inds))[])
  6.411 ns (0 allocations: 0 bytes)
Base.Slice(static(1):100)

julia> @btime ArrayInterface.reduce_tup(ArrayInterface._pick_range, \$(Ref(inds))[])
  2.592 ns (0 allocations: 0 bytes)
Base.Slice(static(1):100)

julia> inds = (Base.OneTo(100), 1:100, 1:UInt(100), Int32(1):Int32(100))
(Base.OneTo(100), 1:100, 0x0000000000000001:0x0000000000000064, 1:100)

julia> @btime reduce(ArrayInterface._pick_range, \$(Ref(inds))[])
  9.048 ns (0 allocations: 0 bytes)
Base.Slice(static(1):100)

julia> @btime ArrayInterface.reduce_tup(ArrayInterface._pick_range, \$(Ref(inds))[])
  2.569 ns (0 allocations: 0 bytes)
Base.Slice(static(1):100)
```
"""
@generated function reduce_tup(f::F, inds::Tuple{Vararg{Any,N}}) where {F,N}
    q = Expr(:block, Expr(:meta, :inline, :propagate_inbounds))
    if N == 1
        push!(q.args, :(inds[1]))
        return q
    end
    syms = Vector{Symbol}(undef, N)
    i = 0
    for n ∈ 1:N
        syms[n] = iₙ = Symbol(:i_, (i += 1))
        push!(q.args, Expr(:(=), iₙ, Expr(:ref, :inds, n)))
    end
    W =  1 << (8sizeof(N) - 2 - leading_zeros(N))
    while W > 0
        _N = length(syms)
        for _ ∈ 2W:W:_N
            for w ∈ 1:W
                new_sym = Symbol(:i_, (i += 1))
                push!(q.args, Expr(:(=), new_sym, Expr(:call, :f, syms[w], syms[w+W])))
                syms[w] = new_sym
            end
            deleteat!(syms, 1+W:2W)
        end
        W >>>= 1
    end
    q
end

@propagate_inbounds function _pick_range(x, y)
    fst = _try_static(static_first(x), static_first(y))
    lst = _try_static(static_last(x), static_last(y))
    return Base.Slice(OptionallyStaticUnitRange(fst, lst))
end

"""
    indices(x, dim) -> AbstractUnitRange{Int}

Given an array `x`, this returns the indices along dimension `dim`.
"""
@inline indices(x, d) = indices(axes(x, d))

"""
    indices(x) -> AbstractUnitRange{Int}

Returns valid indices for the entire length of `x`.
"""
@inline function indices(x)
    inds = eachindex(x)
    if inds isa AbstractUnitRange && eltype(inds) <: Integer
        return Base.Slice(OptionallyStaticUnitRange(inds))
    else
        return inds
    end
end
@inline indices(x::AbstractUnitRange{<:Integer}) = Base.Slice(OptionallyStaticUnitRange(x))

"""
    indices(x::Tuple) -> AbstractUnitRange{Int}

Returns valid indices for the entire length of each array in `x`.
"""
@propagate_inbounds function indices(x::Tuple)
    inds = map(eachindex, x)
    return reduce_tup(_pick_range, inds)
end

"""
    indices(x::Tuple, dim)  -> AbstractUnitRange{Int}

Returns valid indices for each array in `x` along dimension `dim`
"""
@propagate_inbounds function indices(x::Tuple, dim)
    inds = map(x_i -> indices(x_i, dim), x)
    return reduce_tup(_pick_range, inds)
end

"""
    indices(x::Tuple, dim::Tuple) -> AbstractUnitRange{Int}

Returns valid indices given a tuple of arrays `x` and tuple of dimesions for each
respective array (`dim`).
"""
@propagate_inbounds function indices(x::Tuple, dim::Tuple)
    inds = map(indices, x, dim)
    return reduce_tup(_pick_range, inds)
end

"""
    indices(x, dim::Tuple) -> Tuple{Vararg{AbstractUnitRange{Int}}}

Returns valid indices for array `x` along each dimension specified in `dim`.
"""
@inline indices(x, dims::Tuple) = _indices(x, dims)
_indices(x, dims::Tuple) = (indices(x, first(dims)), _indices(x, tail(dims))...)
_indices(x, ::Tuple{}) = ()

