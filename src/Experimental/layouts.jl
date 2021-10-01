
struct Layouted{S<:AccessStyle,P,I,F}
    parent::P
    indices::I
    f::F

    Layouted{S}(p::P, i::I, f::F) where {S,P,I,F} = new{S,P,I,F}(p, i, f)
    Layouted{S}(p, i) where {S} = Layouted{S}(p, i, identity)
end

AccessStyle(::Type{<:Layouted{S}}) where {S} = S()

parent_type(::Type{<:Layouted{S,P}}) where {S,P} = P

Base.parent(x::Layouted) = getfield(x, :parent)

@inline function Base.getindex(x::Layouted{S,P,I}, i) where {S,P,I}
    getfield(x, :f)(@inbounds(parent(x)[getfield(x, :indices)[i]]))
end
@inline function Base.getindex(x::Layouted{S,P,Nothing}, i) where {S,P}
    getfield(x, :f)(@inbounds(parent(x)[i]))
end

@inline function Base.setindex!(x::Layouted{S,P,I}, v, i) where {S,P,I}
    @inbounds(Base.setindex!(parent(x), getfield(x, :f)(v), getfield(x, :indices)[i]))
end
@inline function Base.setindex!(x::Layouted{S,P,Nothing}, v, i) where {S,P}
    @inbounds(Base.setindex!(parent(x), getfield(x, :f)(v), i))
end

"""
    layout(x, access::AccessStyle)

Returns a representation of `x`'s layout given a particular `AccessStyle`.
"""
layout(x, i::CanonicalInt) = Layouted{AccessElement{1}}(x, i)
layout(x, i::AbstractCartesianIndex{N}) where {N} = Layouted{AccessElement{N}}(x, i)
layout(x, i::Tuple{CanonicalInt}) = layout(x, getfield(i, 1))
layout(x, i::Tuple{CanonicalInt,Vararg{CanonicalInt}}) = layout(x, NDIndex(i))
layout(x, i::Tuple{Vararg{Any,N}}) where {N} = Layouted{AccessIndices{N}}(x, i)
layout(x, s::AccessStyle) = Layouted{typeof(s)}(x, nothing)

## Base type ranges
@inline function layout(x::Union{UnitRange,OneTo,StepRange,OptionallyStaticRange}, ::AccessElement{1})
    Layouted{AccessElement{1}}(x, nothing, identity)
end

## Array
layout(x::Array, ::AccessElement{1}) = Layouted{AccessElement{1}}(x, nothing)
@inline layout(x::Array, ::AccessElement) = Layouted{AccessElement{1}}(x, StrideIndex(x))

## ReshapedArray
layout(x::ReshapedArray, ::AccessElement{1}) = Layouted{AccessElement{1}}(parent(x), nothing)
@inline function layout(x::ReshapedArray{T,N}, ::AccessElement{N}) where {T,N}
    Layouted{AccessElement{1}}(x, _to_linear(x))
end

## Transpose/Adjoint{Real}
@inline function layout(x::Union{Transpose{<:Any,<:AbstractMatrix},Adjoint{<:Real,<:AbstractMatrix}}, ::AccessElement{2})
    Layouted{AccessElement{2}}(parent(x), PermutedIndex{2,(2,1),(2,1)}())
end
@inline function layout(x::Union{Transpose{<:Any,<:AbstractVector},Adjoint{<:Real,<:AbstractVector}}, ::AccessElement{2})
    Layouted{AccessElement{1}}(parent(x), PermutedIndex{2,(2,1),(2,)}())
end
@inline function layout(x::Union{Transpose{<:Any,<:AbstractMatrix},Adjoint{<:Real,<:AbstractMatrix}}, ::AccessElement{1})
    Layouted{AccessElement{2}}(parent(x), combined_index(PermutedIndex{2,(2,1),(2,1)}(), _to_cartesian(x)))
end
@inline function layout(x::Union{Transpose{<:Any,<:AbstractVector},Adjoint{<:Real,<:AbstractVector}}, ::AccessElement{1})
    Layouted{AccessElement{1}}(parent(x), nothing)
end

## Adjoint
@inline function layout(x::Adjoint{<:Any,<:AbstractMatrix}, ::AccessElement{2})
    Layouted{AccessElement{2}}(parent(x), PermutedIndex{2,(2,1),(2,1)}(), adjoint)
end
@inline function layout(x::Adjoint{<:Any,<:AbstractVector}, ::AccessElement{2})
    Layouted{AccessElement{1}}(parent(x), PermutedIndex{2,(2,1),(2,)}(), adjoint)
end
@inline function layout(x::Adjoint{<:Any,<:AbstractMatrix}, ::AccessElement{1})
    Layouted{AccessElement{2}}(parent(x), combined_index(PermutedIndex{2,(2,1),(2,1)}(), _to_cartesian(x)), adjoint)
end
@inline function layout(x::Adjoint{<:Any,<:AbstractVector}, ::AccessElement{1})
    Layouted{AccessElement{1}}(parent(x), nothing, adjoint)
end

## PermutedDimsArray
@inline function layout(x::PermutedDimsArray{T,N,I1,I2}, ::AccessElement{1}) where {T,N,I1,I2}
    if N === 1
        return Layouted{AccessElement{1}}(parent(x), nothing)
    else
        return Layouted{AccessElement{N}}(parent(x), combined_index(PermutedIndex{N,I1,I2}(), _to_cartesian(x)))
    end
end
@inline function layout(x::PermutedDimsArray{T,N,I1,I2}, ::AccessElement{N}) where {T,N,I1,I2}
    Layouted{AccessElement{N}}(parent(x), PermutedIndex{N,I1,I2}())
end

## SubArray
@inline function layout(x::Base.FastContiguousSubArray, ::AccessElement{1})
    Layouted{AccessElement{1}}(parent(x), OffsetIndex(getfield(x, :offset1)))
end
@inline function layout(x::Base.FastSubArray, ::AccessElement{1})
    Layouted{AccessElement{1}}(parent(x), LinearSubIndex(getfield(x, :offset1), getfield(x, :stride1)))
end
@inline function layout(x::SubArray{T,N}, ::AccessElement{1}) where {T,N}
    if N === 1
        i = SubIndex{1}(getfield(x, :indices))
        return Layouted{typeof(AccessStyle(i))}(parent(x), i)
    else
        i = SubIndex{N}(getfield(x, :indices))
        return Layouted{typeof(AccessStyle(i))}(parent(x), combined_index(i, _to_cartesian(x)))
    end
end
@inline function layout(x::SubArray{T,N,P,I}, ::AccessElement{N}) where {T,N,P,I}
    Layouted{AccessElement{sum(dynamic(ndims_index(I)))}}(parent(x), SubIndex{N}(getfield(x, :indices)))
end

