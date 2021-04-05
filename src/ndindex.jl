
"""

CartesianIndex(i, j, k...)   -> I
CartesianIndex((i, j, k...)) -> I

Create a multidimensional index I, which can be used for indexing a multidimensional array A. In particular, A[I] is
equivalent to A[i,j,k...]. One can freely mix integer and CartesianIndex indices; for example, A[Ipre, i, Ipost] (where Ipre
and Ipost are CartesianIndex indices and i is an Int) can be a useful expression when writing algorithms that work along a
single dimension of an array of arbitrary dimensionality.

A CartesianIndex is sometimes produced by eachindex, and always when iterating with an explicit CartesianIndices.

Examples
≡≡≡≡≡≡≡≡≡≡

julia> A = reshape(Vector(1:16), (2, 2, 2, 2))
2×2×2×2 Array{Int64, 4}:
[:, :, 1, 1] =
1  3
2  4

[:, :, 2, 1] =
5  7
6  8

[:, :, 1, 2] =
9  11
10  12

[:, :, 2, 2] =
13  15
14  16

julia> A[CartesianIndex((1, 1, 1, 1))]
1

julia> A[CartesianIndex((1, 1, 1, 2))]
9

julia> A[CartesianIndex((1, 1, 2, 1))]
5
"""
struct NDIndex{N,I<:Tuple{Vararg{Any,N}}} <: AbstractCartesianIndex{N}
    index::I

    global _NDIndex(index::Tuple{Vararg{Any,N}}) where {N} = new{N,typeof(index)}(index)

    function NDIndex{N,I}(index::I) where {N,I<:Tuple{Vararg{Integer,N}}}
        for i in index
            (i <: Int) || i <: StaticInt || throw(MethodError("NDIndex does not support values of type $(typeof(i))"))
        end
        return new{N,I}(index)
    end
end

NDIndex{N}() where {N} = new{0,Tuple{}}(())
NDIndex{N}(index::Tuple{Vararg{Any,N}}) where {N} = _ndindex(static(N), _flatten(index...))
NDIndex{N}(index...) where {N} = _ndindex(static(N), _flatten(index...))
NDIndex(index::Tuple) = _NDIndex(_flatten(index...))
NDIndex(index...) = _NDIndex(_flatten(index...))
function _ndindex(n::StaticInt{N}, index::Tuple{Vararg{Integer,M}}) where {N,M}
    M > N && throw(ArgumentError("input tuple of length $M, requested $N"))
    return _NDIndex(_fill_to_length(index, 1, n))
end

_fill_to_length(x::Tuple{Vararg{Any,N}}, n::StaticInt{N}) where {N} = x
@inline function _fill_to_length(x::Tuple{Vararg{Any,M}}, n::StaticInt{N}) where {M,N}
    return _fill_to_length((x..., static(1)))
end

_flatten(i::Integer) = (_int(i),)
_flatten(i::Base.AbstractCartesianIndex) = _flatten(Tuple(i)...)
@inline _flatten(i::Integer, I...) = (_int(i), _flatten(I...)...)
@inline function _flatten(i::Base.AbstractCartesianIndex, I...)
    return (_flatten(Tuple(i)...)..., _flatten(I...)...)
 end
Base.Tuple(index::NDIndex) = index.index

Base.show(io::IO, i::NDIndex) = (print(io, "NDIndex"); show(io, Tuple(i)))

# length
Base.length(::NDIndex{N}) where {N} = N
Base.length(::Type{NDIndex{N}}) where {N} = N

# indexing
@propagate_inbounds getindex(x::NDIndex, i::Integer) = getindex(Tuple(x), i)
@propagate_inbounds Base.getindex(x::NDIndex, i::Integer) = getindex(x, i)
# Base.get(A::AbstractArray, I::CartesianIndex, default) = get(A, I.I, default)
# eltype(::Type{T}) where {T<:CartesianIndex} = eltype(fieldtype(T, :I))

Base.setindex(x::NDIndex, i, j) = NDIndex(Base.setindex(Tuple(x), i, j))

# equality
Base.:(==)(x::NDIndex{N}, y::NDIndex{N}) where N = Tuple(x) == Tuple(y)

# zeros and ones
Base.zero(::NDIndex{N}) where {N} = zero(NDIndex{N})
Base.zero(::Type{NDIndex{N}}) where {N} = _NDIndex(ntuple(_ -> static(0), Val(N)))
Base.oneunit(::NDIndex{N}) where {N} = oneunit(NDIndex{N})
Base.oneunit(::Type{NDIndex{N}}) where {N} = _NDIndex(ntuple(_ -> static(1), Val(N)))

@inline function Base.split(i::NDIndex, V::Val)
    i, j = split(Tuple(i), V)
    return NDIndex(i), NDIndex(j)
end

# arithmetic, min/max
@inline Base.:(-)(i::NDIndex{N}) where {N} = NDIndex{N}(map(-, Tuple(i)))
@inline function Base.:(+)(i1::NDIndex{N}, i2::NDIndex{N}) where {N}
    return NDIndex{N}(map(+, Tuple(i1), Tuple(i2)))
end
@inline function Base.:(-)(i1::NDIndex{N}, i2::NDIndex{N}) where {N}
    return NDIndex{N}(map(-, Tuple(i1), Tuple(i2)))
end
@inline function Base.min(i1::NDIndex{N}, i2::NDIndex{N}) where {N}
    return NDIndex{N}(map(min, Tuple(i1), Tuple(i2)))
end
@inline function Base.max(i1::NDIndex{N}, i2::NDIndex{N}) where {N}
    return NDIndex{N}(map(max, Tuple(i1), Tuple(i2)))
end
@inline Base.:(*)(a::Integer, i::NDIndex{N}) where {N} = NDIndex{N}(map(x->a*x, Tuple(i)))
@inline Base.:(*)(i::NDIndex, a::Integer) = *(a, i)

# comparison
@inline function Base.isless(x::NDIndex{N}, y::NDIndex{N}) where {N}
    return dynamic(_isless(static(false), Tuple(x), Tuple(y)))
end

function _isless(::StaticInt{0}, x::Tuple, y::Tuple)
    return _isless(icmp(last(x), last(y)), Base.front(x), Base.front(y))
end
function _isless(ret::StaticInt{N}, x::Tuple, y::Tuple) where {N}
    return _isless(ret, Base.front(x), Base.front(y))
end
@inline function _isless(ret::Bool, x::Tuple, y::Tuple)
    if ret === 0
        newret = dynamic(icmp(last(x), last(y)))
    else
        newret = ret
    end
    return _isless(newret, Base.front(x), Base.front(y))
end

_isles(::StaticInt{N}, ::Tuple{}, ::Tuple{}) where {N} = static(false)
_isless(::StaticInt{1}, ::Tuple{}, ::Tuple{}) = static(true)
_isless(ret::Int, ::Tuple{}, ::Tuple{}) = ret === 1


icmp(a, b) = _icmp(Static.lt(a, b), a, b)
_icmp(::True, a, b) = static(1)
_icmp(::False, a, b) = __icmp(Static.eq(a, b))
_icmp(x::Bool, a, b) = __icmp(a == b)
__icmp(::True) = static(0)
__icmp(::False) = static(-1)
function __icmp(x::Bool)
    if x
        return 0
    else
        return -1
    end
end

Static.lt(x::NDIndex{N}, y::NDIndex{N}) where {N} = _isless(static(0), Tuple(x), Tuple(y))

_layout(::IndexLinear, x::Tuple) = LinearIndices(x)
_layout(::IndexCartesian, x::Tuple) = CartesianIndices(x)

Base.CartesianIndex(x::NDIndex) = CartesianIndex(Tuple(x))

#  Necessary for compatibility with Base
# In simple cases, we know that we don't need to use axes(A). Optimize those
# until Julia gets smart enough to elide the call on its own:
@inline Base.to_indices(A, I::Tuple{Vararg{Union{Integer,NDIndex}}}) = Base.to_indices(A, (), I)
@inline function Base.to_indices(A, inds, I::Tuple{NDIndex, Vararg{Any}})
    return Base.to_indices(A, inds, (Tuple(I[1])..., tail(I)...))
end
# But for arrays of CartesianIndex, we just skip the appropriate number of inds
@inline function Base.to_indices(A, inds, I::Tuple{AbstractArray{NDIndex{N}}, Vararg{Any}}) where N
    _, indstail = IteratorsMD.split(inds, Val(N))
    return (Base.to_index(A, I[1]), Base.to_indices(A, indstail, tail(I))...)
end

