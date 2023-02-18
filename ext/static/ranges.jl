
ArrayInterface.known_first(::Type{<:OptionallyStaticUnitRange{StaticInt{F}}}) where {F} = F::Int
ArrayInterface.known_first(::Type{<:OptionallyStaticStepRange{StaticInt{F}}}) where {F} = F::Int

ArrayInterface.known_step(::Type{<:OptionallyStaticStepRange{<:Any,StaticInt{S}}}) where {S} = S::Int

ArrayInterface.known_last(::Type{<:OptionallyStaticUnitRange{<:Any,StaticInt{L}}}) where {L} = L::Int
ArrayInterface.known_last(::Type{<:OptionallyStaticStepRange{<:Any,<:Any,StaticInt{L}}}) where {L} = L::Int

"""
    indices(x, dim) -> AbstractUnitRange{Int}

Given an array `x`, this returns the indices along dimension `dim`.
"""
@inline indices(x, d) = indices(static_axes(x, d))

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
    return reduce_tup(static_promote, inds)
end

"""
    indices(x::Tuple, dim)  -> AbstractUnitRange{Int}

Returns valid indices for each array in `x` along dimension `dim`
"""
@propagate_inbounds function indices(x::Tuple, dim)
    inds = map(Base.Fix2(indices, dim), x)
    return reduce_tup(static_promote, inds)
end

"""
    indices(x::Tuple, dim::Tuple) -> AbstractUnitRange{Int}

Returns valid indices given a tuple of arrays `x` and tuple of dimesions for each
respective array (`dim`).
"""
@propagate_inbounds function indices(x::Tuple, dim::Tuple)
    inds = map(indices, x, dim)
    return reduce_tup(static_promote, inds)
end

"""
    indices(x, dim::Tuple) -> Tuple{Vararg{AbstractUnitRange{Int}}}

Returns valid indices for array `x` along each dimension specified in `dim`.
"""
@inline indices(x, dims::Tuple) = _indices(x, dims)
_indices(x, dims::Tuple) = (indices(x, first(dims)), _indices(x, tail(dims))...)
_indices(x, ::Tuple{}) = ()

