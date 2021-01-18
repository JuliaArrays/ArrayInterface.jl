
"""
    known_first(::Type{T})

If `first` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_first(typeof(1:4)))
@test isone(known_first(typeof(Base.OneTo(4))))
"""
known_first(x) = known_first(typeof(x))
function known_first(::Type{T}) where {T}
    if parent_type(T) <: T
        return nothing
    else
        return known_first(parent_type(T))
    end
end
known_first(::Type{Base.OneTo{T}}) where {T} = one(T)

"""
    known_last(::Type{T})

If `last` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_last(typeof(1:4)))
using StaticArrays
@test known_last(typeof(SOneTo(4))) == 4
"""
known_last(x) = known_last(typeof(x))
function known_last(::Type{T}) where {T}
    if parent_type(T) <: T
        return nothing
    else
        return known_last(parent_type(T))
    end
end

"""
    known_step(::Type{T})

If `step` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_step(typeof(1:0.2:4)))
@test isone(known_step(typeof(1:4)))
"""
known_step(x) = known_step(typeof(x))
function known_step(::Type{T}) where {T}
    if parent_type(T) <: T
        return nothing
    else
        return known_step(parent_type(T))
    end
end
known_step(::Type{<:AbstractUnitRange{T}}) where {T} = one(T)

"""
    OptionallyStaticUnitRange(start, stop) <: AbstractUnitRange{Int}

This range permits diverse representations of arrays to communicate common information
about their indices. Each field may be an integer or `Val(<:Integer)` if it is known
at compile time. An `OptionallyStaticUnitRange` is intended to be constructed internally
from other valid indices. Therefore, users should not expect the same checks are used
to ensure construction of a valid `OptionallyStaticUnitRange` as a `UnitRange`.
"""
struct OptionallyStaticUnitRange{F<:Integer,L<:Integer} <: AbstractUnitRange{Int}
    start::F
    stop::L

    function OptionallyStaticUnitRange(start, stop)
        if eltype(start) <: Int
            if eltype(stop) <: Int
                return new{typeof(start),typeof(stop)}(start, stop)
            else
                return OptionallyStaticUnitRange(start, Int(stop))
            end
        else
            return OptionallyStaticUnitRange(Int(start), stop)
        end
    end

    function OptionallyStaticUnitRange{F,L}(x::AbstractRange) where {F,L}
        if step(x) == 1
            return OptionallyStaticUnitRange(static_first(x), static_last(x))
        else
            throw(ArgumentError("step must be 1, got $(step(x))"))
        end
    end

    function OptionallyStaticUnitRange(x::AbstractRange)
        if step(x) == 1
            return OptionallyStaticUnitRange(static_first(x), static_last(x))
        else
            throw(ArgumentError("step must be 1, got $(step(x))"))
        end
    end
end

Base.first(r::OptionallyStaticUnitRange) = r.start
Base.step(::OptionallyStaticUnitRange) = StaticInt(1)
Base.last(r::OptionallyStaticUnitRange) = r.stop

known_first(::Type{<:OptionallyStaticUnitRange{StaticInt{F}}}) where {F} = F
known_step(::Type{<:OptionallyStaticUnitRange}) = 1
known_last(::Type{<:OptionallyStaticUnitRange{<:Any,StaticInt{L}}}) where {L} = L

"""
    OptionallyStaticStepRange(start, step, stop) <: OrdinalRange{Int,Int}

Similarly to [`OptionallyStaticUnitRange`](@ref), `OptionallyStaticStepRange` permits
a combination of static and standard primitive `Int`s to construct a range. It
specifically enables the use of ranges without a step size of 1. It may be constructed
through the use of `OptionallyStaticStepRange` directly or using static integers with
the range operator (i.e., `:`).

```julia
julia> using ArrayInterface

julia> x = ArrayInterface.StaticInt(2);

julia> x:x:10
ArrayInterface.StaticInt{2}():ArrayInterface.StaticInt{2}():10

julia> ArrayInterface.OptionallyStaticStepRange(x, x, 10)
ArrayInterface.StaticInt{2}():ArrayInterface.StaticInt{2}():10
```
"""
struct OptionallyStaticStepRange{F<:Integer,S<:Integer,L<:Integer} <: OrdinalRange{Int,Int}
    start::F
    step::S
    stop::L

    function OptionallyStaticStepRange(start, step, stop)
        if eltype(start) <: Int
            if eltype(stop) <: Int
                lst = _steprange_last(start, step, stop)
                return new{typeof(start),typeof(step),typeof(lst)}(start, step, lst)
            else
                return OptionallyStaticStepRange(start, step, Int(stop))
            end
        else
            return OptionallyStaticStepRange(Int(start), step, stop)
        end
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
Base.first(r::OptionallyStaticStepRange) = r.start
Base.step(r::OptionallyStaticStepRange) = r.step
Base.last(r::OptionallyStaticStepRange) = r.stop

known_first(::Type{<:OptionallyStaticStepRange{StaticInt{F}}}) where {F} = F
known_step(::Type{<:OptionallyStaticStepRange{<:Any,StaticInt{S}}}) where {S} = S
known_last(::Type{<:OptionallyStaticStepRange{<:Any,<:Any,StaticInt{L}}}) where {L} = L

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
function Base.:(:)(::StaticInt{F}, ::StaticInt{1}, ::StaticInt{L}) where {F,L}
    return OptionallyStaticUnitRange(StaticInt(F), StaticInt(L))
end
function Base.:(:)(start::Integer, ::StaticInt{1}, ::StaticInt{L}) where {L}
    return OptionallyStaticUnitRange(start, StaticInt(L))
end
function Base.:(:)(::StaticInt{F}, ::StaticInt{1}, stop::Integer) where {F}
    return OptionallyStaticUnitRange(StaticInt(F), stop)
end
function Base.:(:)(start::Integer, ::StaticInt{1}, stop::Integer)
    return OptionallyStaticUnitRange(start, stop)
end

function Base.isempty(r::OptionallyStaticUnitRange)
    if known_first(r) === oneunit(eltype(r))
        return unsafe_isempty_one_to(last(r))
    else
        return unsafe_isempty_unit_range(first(r), last(r))
    end
end

function Base.isempty(r::OptionallyStaticStepRange)
    return (r.start != r.stop) & ((r.step > zero(r.step)) != (r.stop > r.start))
end

unsafe_isempty_one_to(lst) = lst <= zero(lst)
unsafe_isempty_unit_range(fst, lst) = fst > lst

unsafe_length_one_to(lst::Int) = lst
unsafe_length_one_to(::StaticInt{L}) where {L} = L

# TODO this should probably be renamed because the point is that it is safe
@inline function unsafe_length_step_range(start::Int, step::Int, stop::Int)
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

@propagate_inbounds function Base.getindex(
    r::OptionallyStaticUnitRange,
    s::AbstractUnitRange{<:Integer},
)
    @boundscheck checkbounds(r, s)
    f = static_first(r)
    fnew = f - one(f)
    return (fnew+static_first(s)):(fnew+static_last(s))
end

@propagate_inbounds function Base.getindex(r::OptionallyStaticUnitRange, i::Integer)
    if known_first(r) === oneunit(eltype(r))
        return get_index_one_to(r, i)
    else
        return get_index_unit_range(r, i)
    end
end

@inline function get_index_one_to(r, i)
    @boundscheck if ((i < 1) || (i > last(r)))
        throw(BoundsError(r, i))
    end
    return convert(eltype(r), i)
end

@inline function get_index_unit_range(r, i)
    val = first(r) + (i - 1)
    @boundscheck if (i < 1) || val > last(r)
        throw(BoundsError(r, i))
    end
    return convert(eltype(r), val)
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

###
### length
###
@inline function known_length(::Type{T}) where {T<:OptionallyStaticUnitRange}
    fst = known_first(T)
    lst = known_last(T)
    if fst === nothing || lst === nothing
        return nothing
    else
        if fst === oneunit(eltype(T))
            return unsafe_length_one_to(lst)
        else
            return unsafe_length_unit_range(fst, lst)
        end
    end
end

@inline function known_length(::Type{T}) where {T<:OptionallyStaticStepRange}
    fst = known_first(T)
    stp = known_step(T)
    lst = known_last(T)
    if fst === nothing || stp === nothing || lst === nothing
        return nothing
    else
        if stp === 1
            if fst === oneunit(eltype(T))
                return unsafe_length_one_to(lst)
            else
                return unsafe_length_unit_range(fst, lst)
            end
        else
            return unsafe_length_step_range(fst, stp, lst)
        end
    end
end

function Base.length(r::OptionallyStaticUnitRange)
    if isempty(r)
        return 0
    else
        if known_first(r) === 1
            return unsafe_length_one_to(last(r))
        else
            return unsafe_length_unit_range(first(r), last(r))
        end
    end
end

function Base.length(r::OptionallyStaticStepRange)
    if isempty(r)
        return 0
    else
        if known_step(r) === 1
            if known_first(r) === 1
                return unsafe_length_one_to(last(r))
            else
                return unsafe_length_unit_range(first(r), last(r))
            end
        else
            return unsafe_length_step_range(Int(first(r)), Int(step(r)), Int(last(r)))
        end
    end
end

unsafe_length_unit_range(start::Integer, stop::Integer) = Int((stop - start) + 1)

function Base.AbstractUnitRange{T}(r::OptionallyStaticUnitRange) where {T}
    if T <: Int
        return r
    else
        if known_first(r) === 1 && T <: Integer
            return OneTo{T}(last(r))
        else
            return UnitRange{T}(first(r), last(r))
        end
    end
end

const OptionallyStaticRange = Union{<:OptionallyStaticUnitRange,<:OptionallyStaticStepRange}

Base.eachindex(r::OptionallyStaticRange) = r
@inline Base.iterate(r::OptionallyStaticRange) = (fi = Int(first(r)); (fi, fi))


Base.to_shape(x::OptionallyStaticRange) = length(x)
Base.to_shape(x::Slice{T}) where {T<:OptionallyStaticRange} = length(x)

@inline function Base.axes(S::Slice{T}) where {T<:OptionallyStaticRange}
    if known_first(T) === 1 && known_step(T) === 1
        return (S.indices,)
    else
        return (Base.IdentityUnitRange(S.indices),)
    end
end

@inline function Base.axes1(S::Slice{T}) where {T<:OptionallyStaticRange}
    if known_first(T) === 1 && known_step(T) === 1
        return S.indices
    else
        return Base.IdentityUnitRange(S.indices)
    end
end

Base.:(-)(r::OptionallyStaticRange) = -static_first(r):-static_step(r):-static_last(r)

"""
    indices(x[, d])

Given an array `x`, this returns the indices along dimension `d`. If `x` is a tuple
of arrays, then the indices corresponding to dimension `d` of all arrays in `x` are
returned. If any indices are not equal along dimension `d`, an error is thrown. A
tuple may be used to specify a different dimension for each array. If `d` is not
specified, then the indices for visiting each index of `x` are returned.
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
julia> using ArrayInterface, BenchmarkTools

julia> inds = (Base.OneTo(100), 1:100, 1:ArrayInterface.StaticInt(100))
(Base.OneTo(100), 1:100, 1:Static(100))

julia> @btime reduce(ArrayInterface._pick_range, \$(Ref(inds))[])
  6.405 ns (0 allocations: 0 bytes)
Base.Slice(Static(1):Static(100))

julia> @btime ArrayInterface.reduce_tup(ArrayInterface._pick_range, \$(Ref(inds))[])
  2.570 ns (0 allocations: 0 bytes)
Base.Slice(Static(1):Static(100))

julia> inds = (Base.OneTo(100), 1:100, 1:UInt(100))
(Base.OneTo(100), 1:100, 0x0000000000000001:0x0000000000000064)

julia> @btime reduce(ArrayInterface._pick_range, \$(Ref(inds))[])
  6.411 ns (0 allocations: 0 bytes)
Base.Slice(Static(1):100)

julia> @btime ArrayInterface.reduce_tup(ArrayInterface._pick_range, \$(Ref(inds))[])
  2.592 ns (0 allocations: 0 bytes)
Base.Slice(Static(1):100)

julia> inds = (Base.OneTo(100), 1:100, 1:UInt(100), Int32(1):Int32(100))
(Base.OneTo(100), 1:100, 0x0000000000000001:0x0000000000000064, 1:100)

julia> @btime reduce(ArrayInterface._pick_range, \$(Ref(inds))[])
  9.048 ns (0 allocations: 0 bytes)
Base.Slice(Static(1):100)

julia> @btime ArrayInterface.reduce_tup(ArrayInterface._pick_range, \$(Ref(inds))[])
  2.569 ns (0 allocations: 0 bytes)
Base.Slice(Static(1):100)
```
"""
@generated function reduce_tup(f::F, inds::Tuple{Vararg{Any,N}}) where {F,N}
    q = Expr(:block, Expr(:meta, :inline))
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

function indices(x::Tuple)
    inds = map(eachindex, x)
    return reduce_tup(_pick_range, inds)
end

@inline indices(x, d) = indices(axes(x, d))

@inline function indices(x::Tuple{Vararg{Any,N}}, dim) where {N}
    inds = map(x_i -> indices(x_i, dim), x)
    return reduce_tup(_pick_range, inds)
end

@inline function indices(x::Tuple{Vararg{Any,N}}, dim::Tuple{Vararg{Any,N}}) where {N}
    inds = map(indices, x, dim)
    return reduce_tup(_pick_range, inds)
end

@inline function _pick_range(x, y)
    fst = _try_static(static_first(x), static_first(y))
    lst = _try_static(static_last(x), static_last(y))
    return Base.Slice(OptionallyStaticUnitRange(fst, lst))
end

function Base.show(io::IO, r::OptionallyStaticRange)
    print(io, first(r))
    if known_step(r) === 1
        print(io, ":")
    else
        print(io, ":")
        print(io, step(r))
        print(io, ":")
    end
    print(io, last(r))
end
