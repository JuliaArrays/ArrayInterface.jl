
"""
known_first(::Type{T})

If `first` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_first(typeof(1:4)))
@test isone(known_first(typeof(Base.OneTo(4))))
"""
known_first(x) = known_first(typeof(x))
known_first(::Type{T}) where {T} = nothing
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
known_last(::Type{T}) where {T} = nothing

"""
known_step(::Type{T})

If `step` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_step(typeof(1:0.2:4)))
@test isone(known_step(typeof(1:4)))
"""
known_step(x) = known_step(typeof(x))
known_step(::Type{T}) where {T} = nothing
known_step(::Type{<:AbstractUnitRange{T}}) where {T} = one(T)

_eltype(::Type{T}) where {T} = T
_eltype(::Type{Val{V}}) where {V} = typeof(V)

struct OptionallyStaticRange{T<:Integer,F,S,L} <: OrdinalRange{T,T}
  start::F
  step::S
  stop::L

  function OptionallyStaticRange(start::F, step::S, stop::L) where {F,S,L}
      T = promote_type(_eltype(F), _eltype(S), eltype(L))
      return new{T,F,S,L}(start, step, stop)
  end
end

Base.first(r::OptionallyStaticRange{<:Any,Val{F}}) where {F} = F
Base.first(r::OptionallyStaticRange{<:Any,<:Any}) = getfield(r, :start)

Base.step(r::OptionallyStaticRange{<:Any,<:Any,Val{S}}) where {S} = S
Base.step(r::OptionallyStaticRange{<:Any,<:Any,<:Any}) = getfield(r, :step)

Base.last(r::OptionallyStaticRange{<:Any,<:Any,<:Any,Val{L}}) where {L} = L
Base.last(r::OptionallyStaticRange{<:Any,<:Any,<:Any,<:Any}) = getfield(r, :stop)

ArrayInterface.known_first(::OptionallyStaticRange{<:Any,Val{F}}) where {F} = F
ArrayInterface.known_step(::OptionallyStaticRange{<:Any,<:Any,Val{S}}) where {S} =S
ArrayInterface.known_last(::OptionallyStaticRange{<:Any,<:Any,<:Any,Val{L}}) where {L} = L

function Base.isempty(r::OptionallyStaticRange)
    return (first(r) != last(r)) & ((step(r) > zero(step(r))) != (last(r) > first(r)))
end

@inline function Base.length(r::OptionallyStaticRange{T}) where {T}
    if isempty(r)
        return zero(T)
    else
        if known_step(r) === oneunit(T)
            if known_first(r) === oneunit(T)
                return last(r)
            else
                return last(r) - first(r) + step(r)
            end
        else
            return Integer(div((last(r) - first(r)) + step(r), step(r)))
        end
    end
end


isempty(r::StepRange) =
    (r.start != r.stop) & ((r.step > zero(r.step)) != (r.stop > r.start))
isempty(r::AbstractUnitRange) = first(r) > last(r)
isempty(r::StepRangeLen) = length(r) == 0
isempty(r::LinRange) = length(r) == 0

# add methods to support ArrayInterface

_try_static(x, y) = Val(x)
_try_static(::Nothing, y) = Val(y)
_try_static(x, ::Nothing) = Val(x)
_try_static(::Nothing, ::Nothing) = nothing

@inline function _pick_range(x, y)
    fst = _try_static(known_first(x), known_first(y))
    fst = fst === nothing ? first(x) : fst

    st = _try_static(known_step(x), known_step(y))
    st = st === nothing ? step(x) : st

    lst = _try_static(known_last(x), known_last(y))
    lst = lst === nothing ? last(x) : lst
    return OptionallyStaticRange(fst, st, lst)
end

"""
    indices(x[, d]) -> AbstractRange

Given an array `x`, this returns the indices along dimension `d`. If `x` is a tuple
of arrays then the indices corresponding to dimension `d` of all arrays in `x` are
returned. If any indices are not equal along dimension `d` an error is thrown. A
tuple may be used to specify a different dimension for each array. If `d` is not
specified then indices for visiting each index of `x` is returned.
"""
@inline indices(x) = eachindex(x)

indices(x, d) = indices(axes(x, d))

@inline function indices(x::NTuple{N,<:Any}, dim) where {N}
  inds = map(x_i -> indices(x_i, dim), x)
  @assert all(isequal(first(inds)), Base.tail(inds)) "Not all specified axes are equal: $inds"
  return reduce(_pick_range, inds)
end

@inline function indices(x::NTuple{N,<:Any}, dim::NTuple{N,<:Any}) where {N}
  inds = map(indices, x, dim)
  @assert all(isequal(first(inds)), Base.tail(inds)) "Not all specified axes are equal: $inds"
  return reduce(_pick_range, inds)
end


