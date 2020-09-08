
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
known_first(::Type{T}) where {T<:Base.Slice} = known_first(parent_type(T))

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
known_last(::Type{T}) where {T<:Base.Slice} = known_last(parent_type(T))

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

# add methods to support ArrayInterface

_get(x) = x
_get(::Val{V}) where {V} = V
_get(::Static{V}) where {V} = V
_get(::Type{Static{V}}) where {V} = V
_convert(::Type{T}, x) where {T} = convert(T, x)
_convert(::Type{T}, ::Val{V}) where {T,V} = Val(convert(T, V))

"""
    OptionallyStaticUnitRange{T<:Integer}(start, stop) <: OrdinalRange{T,T}

This range permits diverse representations of arrays to comunicate common information 
about their indices. Each field may be an integer or `Val(<:Integer)` if it is known
at compile time. An `OptionallyStaticUnitRange` is intended to be constructed internally
from other valid indices. Therefore, users should not expect the same checks are used
to ensure construction of a valid `OptionallyStaticUnitRange` as a `UnitRange`.
"""
struct OptionallyStaticUnitRange{T <: Integer, F <: Integer, L <: Integer} <: AbstractUnitRange{T}
  start::F
  stop::L

  function OptionallyStaticUnitRange{T}(start, stop) where {T<:Real}
    if _get(start) isa T
      if _get(stop) isa T
        return new{T,typeof(start),typeof(stop)}(start, stop)
      else
        return OptionallyStaticUnitRange{T}(start, _convert(T, stop))
      end
    else
      return OptionallyStaticUnitRange{T}(_convert(T, start), stop)
    end
  end

  function OptionallyStaticUnitRange(start, stop)
    T = promote_type(typeof(_get(start)), typeof(_get(stop)))
    return OptionallyStaticUnitRange{T}(start, stop)
  end

  function OptionallyStaticUnitRange(x::AbstractRange)
    if step(x) == 1
      fst = static_first(x)
      lst = static_last(x)
      return OptionallyStaticUnitRange(fst, lst)
    else
        throw(ArgumentError("step must be 1, got $(step(r))"))
    end
  end
end

Base.:(:)(L, ::Static{U}) where {U} = OptionallyStaticUnitRange(L, Static(U))
Base.:(:)(::Static{L}, U) where {L} = OptionallyStaticUnitRange(Static(L), U)
Base.:(:)(::Static{L}, ::Static{U}) where {L,U} = OptionallyStaticUnitRange(Static(L), Static(U))

Base.first(r::OptionallyStaticUnitRange) = r.start
Base.step(r::OptionallyStaticUnitRange{T}) where {T} = oneunit(T)
Base.last(r::OptionallyStaticUnitRange) = r.stop

known_first(::Type{<:OptionallyStaticUnitRange{<:Any,Static{F}}}) where {F} = F
known_step(::Type{<:OptionallyStaticUnitRange{T}}) where {T} = one(T)
known_last(::Type{<:OptionallyStaticUnitRange{<:Any,<:Any,Static{L}}}) where {L} = L

function Base.isempty(r::OptionallyStaticUnitRange)
  if known_first(r) === oneunit(eltype(r))
    return unsafe_isempty_one_to(last(r))
  else
    return unsafe_isempty_unit_range(first(r), last(r))
  end
end

unsafe_isempty_one_to(lst) = lst <= zero(lst)
unsafe_isempty_unit_range(fst, lst) = fst > lst

unsafe_isempty_unit_range(fst::T, lst::T) where {T} = Integer(lst - fst + one(T))

unsafe_length_one_to(lst::T) where {T<:Int} = T(lst)
unsafe_length_one_to(lst::T) where {T} = Integer(lst - zero(lst))

Base.@propagate_inbounds function Base.getindex(r::OptionallyStaticUnitRange, i::Integer)
  if known_first(r) === oneunit(r)
    return get_index_one_to(r, i)
  else
    return get_index_unit_range(r, i)
  end
end

@inline function get_index_one_to(r, i)
  @boundscheck if ((i > 0) & (i <= last(r)))
    throw(BoundsError(r, i))
  end
  return convert(eltype(r), i)
end

@inline function get_index_unit_range(r, i)
  val = first(r) + (i - 1)
  @boundscheck if i > 0 && val <= last(r) && val >= first(r)
    throw(BoundsError(r, i))
  end
  return convert(eltype(r), val)
end

@inline _try_static(::Static{N}, ::Static{N}) where {N} = Static{N}()
function _try_static(::Static{N}, x) where {N}
    @assert N == x "Unequal Indices: Static{$N}() != x == $x"
    Static{N}()
end
function _try_static(x, ::Static{N}) where {N}
    @assert N == x "Unequal Indices: x == $x != Static{$N}()"
    Static{N}()
end
function _try_static(x, y)
    @assert x == y "Unequal Indicess: x == $x != $y == y"
    x
end

###
### length
###
@inline function known_length(::Type{T}) where {T<:AbstractUnitRange}
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

function Base.length(r::OptionallyStaticUnitRange{T}) where {T}
  if isempty(r)
    return zero(T)
  else
    if known_one(r) === one(T)
      return unsafe_length_one_to(last(r))
    else
      return unsafe_length_unit_range(first(r), last(r))
    end
  end
end

function unsafe_length_unit_range(fst::T, lst::T) where {T<:Union{Int,Int64,Int128}}
  return Base.checked_add(Base.checked_sub(lst, fst), one(T))
end
function unsafe_length_unit_range(fst::T, lst::T) where {T<:Union{UInt,UInt64,UInt128}}
  return Base.checked_add(lst - fst, one(T))
end

"""
    indices(x[, d])

Given an array `x`, this returns the indices along dimension `d`. If `x` is a tuple
of arrays then the indices corresponding to dimension `d` of all arrays in `x` are
returned. If any indices are not equal along dimension `d` an error is thrown. A
tuple may be used to specify a different dimension for each array. If `d` is not
specified then indices for visiting each index of `x` is returned.
"""
@inline function indices(x)
  inds = eachindex(x)
  if inds isa AbstractUnitRange#{<:Integer} # prevents inference
    return Base.Slice(OptionallyStaticUnitRange(inds))
  else
    return inds
  end
end

function indices(x::Tuple)
  inds = map(eachindex, x)
  return reduce(_pick_range, inds)
end

@inline indices(x, d) = indices(axes(x, d))

@inline function indices(x::Tuple{Vararg{Any,N}}, dim) where {N}
  inds = map(x_i -> indices(x_i, dim), x)
  return reduce(_pick_range, inds)
end

@inline function indices(x::Tuple{Vararg{Any,N}}, dim::Tuple{Vararg{Any,N}}) where {N}
  inds = map(indices, x, dim)
  return reduce(_pick_range, inds)
end

@inline function _pick_range(x, y)
  fst = _try_static(static_first(x), static_first(y))
  lst = _try_static(static_last(x), static_last(y))
  return Base.Slice(OptionallyStaticUnitRange(fst, lst))
end

