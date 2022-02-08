
"""
    Size(s::Tuple{Vararg{Union{Int,StaticInt}})
    Size(A) -> Size(size(A))

Type that represents statically sized dimensions as `StaticInt`s.
"""
struct Size{S<:Tuple}
    size::S

    Size{S}(s::Tuple{Vararg{<:CanonicalInt}}) where {S} = new{S}(s::S)
    Size(s::Tuple{Vararg{<:CanonicalInt}}) = Size{typeof(s)}(s)
end

"""
    Length(x::Union{Int,StaticInt})
    Length(A) = Length(length(A))

Type that represents statically sized dimensions as `StaticInt`s.
"""
const Length{L} = Size{Tuple{L}}
Length(x::CanonicalInt) = Size((x,))
@inline function Length(x)
    len = known_length(x)
    if len === missing
        return Length(length(x))
    else
        return Length(static(len))
    end
end

Base.ndims(@nospecialize(s::Size)) = ndims(typeof(s))
Base.ndims(::Type{<:Size{S}}) where {S} = known_length(S)
Base.size(s::Size{Tuple{Vararg{Int}}}) = getfield(s, :size)
Base.size(s::Size) = map(Int, s.size)
function Base.size(s::Size{S}, dim::CanonicalInt) where {S}
    if dim > known_length(S)
        return 1
    else
        return Int(getfield(s.size, Int(dim)))
    end
end

Base.:(==)(x::Size, y::Size) = getfield(x, :size) == getfield(y, :size)

static_length(x::Length) = getfield(getfield(x, :size), 1)
static_length(x::Size) = prod(getfield(x, :size))
Base.length(x::Size) = Int(static_length(x))

Base.show(io::IO, ::MIME"text/plain", @nospecialize(x::Size)) = print(io, "Size($(x.size))")

# default constructors
Size(s::Size) = s
Size(a::A) where {A} = Size(_maybe_size(Base.IteratorSize(A), a))
_maybe_size(::Base.HasShape{N}, a::A) where {N,A} = map(static_length, axes(a))
_maybe_size(::Base.HasLength, a::A) where {A} = (static_length(a),)

# type specific constructors
Size(x::SubArray) = Size(eachop(_sub_size, to_parent_dims(x), x.indices))
_sub_size(x::Tuple, ::StaticInt{dim}) where {dim} = static_length(getfield(x, dim))
@inline Size(A::VecAdjTrans) = Size((One(), static_length(parent(A))))
@inline function Size(A::MatAdjTrans)
    Size(permute(getfield(Size(parent(A)), :size), (static(2), static(1))))
end
Size(A::Union{Array,ReshapedArray}) = Size(Base.size(A))
@inline function Size(A::PermutedDimsArray{T,N,I1}) where {T,N,I1}
    Size(permute(getfield(Size(parent(A)), :size), static(I1)))
end
Size(A::AbstractRange) = Size((static_length(A),))
Size(x::Base.Generator) = Size(getfield(x, :iter))
Size(x::Iterators.Reverse) = Size(getfield(x, :itr))
Size(x::Iterators.Enumerate) = Size(getfield(x, :itr))
Size(x::Iterators.Accumulate) = Size(getfield(x, :itr))
Size(x::Iterators.Pairs) = Size(getfield(x, :itr))
@inline function Size(x::Iterators.ProductIterator)
    Size(eachop(_sub_size, nstatic(Val(ndims(x))), getfield(x, :iterators)))
end
Size(x::Iterators.Zip) = Size(Static.reduce_tup(promote_shape, map(size, getfield(x, :is))))

function Size(a::ReinterpretArray{T,N,S,A}) where {T,N,S,A}
    if _is_reshaped(typeof(a))
        if sizeof(S) === sizeof(T)
            return Size(parent(a))
        elseif sizeof(S) > sizeof(T)
            return Size((static(div(sizeof(S), sizeof(T))), getfield(Size(parent(a)), :size)...))
        else
            return Size(tail(getfield(Size(parent(a)), :size)))
        end
    else
        psize = getfield(Size(parent(a)), :size)
        return Size((div(first(psize) * static(sizeof(S)), static(sizeof(T))), tail(psize)...,))
    end
end

## size of individual dimensions
Size(x, dim) = Size(x, to_dims(x, dim))
function Size(x, dim::Int)
    sz = known_size(x, dim)
    if sz === missing
        return Length(Int(getfield(getfield(Size(x), :size), dim)))
    else
        return Length(Int(sz))
    end
end
@inline function Size(x, ::StaticInt{dim}) where {dim}
    sz = known_size(x, dim)
    if sz === missing
        return Length(getfield(getfield(Size(x), :size), dim))
    else
        return Length(static(sz))
    end
end

"""
    known_size(::Type{T}) -> Tuple
    known_size(::Type{T}, dim) -> Union{Int,Missing}

Returns the size of each dimension of `A` or along dimension `dim` of `A` that is known at
compile time. If a dimension does not have a known size along a dimension then `missing` is
returned in its position.
"""
known_size(x) = known_size(typeof(x))
function known_size(::Type{T}) where {T<:AbstractRange}
    (_range_length(known_first(T), known_step(T), known_last(T)),)
end
known_size(::Type{<:Size{S}}) where {S} = known(S)
known_size(::Type{<:Base.Generator{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Reverse{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Enumerate{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Accumulate{<:Any,I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Pairs{<:Any,<:Any,I}}) where {I} = known_size(I)
@inline function known_size(::Type{<:Iterators.ProductIterator{T}}) where {T}
    eachop(_known_size, nstatic(Val(known_length(T))), T)
end

# 1. `Zip` doesn't check that its collections are compatible (same size) at construction,
#   but we assume as much b/c otherwise it will error while iterating. So we promote to the
#   known size if matching a `Missing` and `Int` size.
# 2. `promote_shape(::Tuple{Vararg{CanonicalInt}}, ::Tuple{Vararg{CanonicalInt}})` promotes
#   trailing dimensions (which must be of size 1), to `static(1)`. We want to stick to
#   `Missing` and `Int` types, so we do one last pass to ensure everything is dynamic
@inline function known_size(::Type{<:Iterators.Zip{T}}) where {T}
    dynamic(reduce_tup(_promote_shape, eachop(_unzip_size, nstatic(Val(known_length(T))), T)))
end
_unzip_size(::Type{T}, n::StaticInt{N}) where {T,N} = known_size(field_type(T, n))

known_size(::Type{T}) where {T} = _maybe_known_size(Base.IteratorSize(T), T)
function _maybe_known_size(::Base.HasShape{N}, ::Type{T}) where {N,T}
    eachop(_known_size, nstatic(Val(N)), axes_types(T))
end
_maybe_known_size(::Base.IteratorSize, ::Type{T}) where {T} = (known_length(T),)
_known_size(::Type{T}, dim::StaticInt) where {T} = known_length(field_type(T, dim))
@inline known_size(x, dim) = known_size(typeof(x), dim)
@inline known_size(::Type{T}, dim) where {T} = known_size(T, to_dims(T, dim))
@inline function known_size(::Type{T}, dim::CanonicalInt) where {T}
    if ndims(T) < dim
        return 1
    else
        return known_size(T)[dim]
    end
end

size(x) = getfield(Size(x), :size)
size(x, dim) = static_length(Size(x, dim))

