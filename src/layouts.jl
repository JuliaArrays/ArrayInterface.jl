
struct NDIndex{N,I<:Tuple{Vararg{Any,N}}} <: AbstractCartesianIndex{N}
    index::I

    global _NDIndex(index::Tuple{Vararg{Any,N}}) where {N} = new{N,typeof(index)}(index)

    function NDIndex{N,I}(index::I) where {N,I<:Tuple{Vararg{Integer,N}}}
        for i in index
            eltype(i) <: Int || throw(MethodError("NDIndex does not support values of type $(typeof(i))"))
        end
        return new{N,I}(index)
    end
end

NDIndex{N}() where {N} = new{0,Tuple{}}(())
NDIndex{N}(index::Tuple{Vararg{Any,N}}) where {N} = _ndindex(static(N), _flatten(index...))
NDIndex{N}(index...) where {N} = _ndindex(static(N), _flatten(index...))
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

# conversions
Base.convert(::Type{T}, i::NDIndex{1}) where {T<:Number} = convert(T, first(i))
Base.convert(::Type{T}, i::NDIndex) where {T<:Tuple} = convert(T, Tuple(i))

# hashing
const cartindexhash_seed = UInt == UInt64 ? 0xd60ca92f8284b8b0 : 0xf2ea7c2e
function Base.hash(i::NDIndex, h::UInt)
    h += cartindexhash_seed
    for i in Tuple(i)
        h = hash(i, h)
    end
    return h
end

struct Indices{N,I<:Tuple{Vararg{Any,N}}} <: AbstractArray2{NDIndex{N},N}
    indices::I

    _Indices(inds::Tuple{Vararg{Any,N}}) where {N} = new{N,typeof(inds)}(inds)
    function Indices(inds::Tuple)
        for i in inds
            eltype(i) <: Int || throw(ArgumentError("all indices must have Int eltypes"))
        end
        return _Indices(inds)
    end
end


""" SubAxis(axis, subindices) """
struct SubIndices{P,I} <: AbstractArray2{Int,1}
    parent::P
    indices::I
end

struct CSCIndices <: AbstractArray2{Int,2}
    colptr::Vector{Int}
    rowval::Vector{Int}
end

struct DiagonalIndices{T,P<:AbstractArray{T,1},UL} <: AbstractArray2{T,2}
    parent::P
    uplo::UL
end

struct TriangularIndices{T,P<:AbstractArray{T,1},UL} <: AbstractArray2{T,2}
    parent::P
    uplo::UL
end

Base.parent(x::DiagonalIndices) = getifeld(x, :parent)
Base.parent(x::TriangularIndices) = getifeld(x, :parent)

layout(x) = layout(IndexStyle(x), axes(x))
layout(::IndexLinear, axes::Tuple{Vararg{Any,1}}) = first(axes)
layout(::IndexLinear, axes::Tuple{Vararg{Any,N}}) where {N} = LinearIndices(axes)
layout(::IndexCartesian, axes::Tuple{Vararg{Any,N}}) where {N} = CartesianIndices(axes)
layout(x::SymTridiagonal) = DiagonalIndices(layout(x.dv), static(:UL))
layout(x::Bidiagonal) = DiagonalIndices(layout(x.dv), Symbol(x.uplo))
layout(x::Diagonal) = DiagonalIndices(layout(x.dv), static(Symbol("")))

#=
LayoutStyle(::Type{<:AbstractUnitRange{Int}}) = IndexLayout()
LayoutStyle(::Type{<:Diag{T,V}}) where {T,V} = DiagonalLayout{LayoutStyle(V)}()
LayoutStyle(::Type{<:UpTri{T,M}}) where {T,M} = UpperTriangularLayout{LayoutStyle(M)}()
LayoutStyle(::Type{<:LoTri{T,M}}) where {T,M} = LowerTriangularLayout{LayoutStyle(M)}()
LayoutStyle(::Type{<:AbstractSparseArray}) = SparseLayout()


index + offset_x - offset_y

offset_x
offset_y
index


remove_offsets(A, i::Tuple)




function to_layout(dst, src, i::NDIndex) end
function to_layout(dst, src, i::Integer) end

layout


    if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
    r1 = Int(getcolptr(A)[i1])
    r2 = Int(getcolptr(A)[i1+1]-1)
    (r1 > r2) && return zero(T)
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Forward)
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? zero(T) : nonzeros(A)[r1]


layout(x::

=#

