
"""
    size(A) -> Tuple
    size(A, dim) -> Union{Int,StaticInt}

Returns the size of each dimension of `A` or along dimension `dim` of `A`. If the size of
any axes are known at compile time, these should be returned as `Static` numbers. Otherwise,
`ArrayInterface.size(A)` is identical to `Base.size(A)`

```julia
julia> using StaticArrays, ArrayInterface

julia> A = @SMatrix rand(3,4);

julia> ArrayInterface.size(A)
(static(3), static(4))
```
"""
size(a::A) where {A} = _maybe_size(Base.IteratorSize(A), a)
_maybe_size(::Base.HasShape{N}, a::A) where {N,A} = map(static_length, axes(a))
_maybe_size(::Base.HasLength, a::A) where {A} = (static_length(a),)
size(x::SubArray) = eachop(_sub_size, to_parent_dims(x), x.indices)
_sub_size(x::Tuple, ::StaticInt{dim}) where {dim} = static_length(getfield(x, dim))
@inline size(B::VecAdjTrans) = (One(), length(parent(B)))
@inline size(B::MatAdjTrans) = permute(size(parent(B)), to_parent_dims(B))
@inline function size(B::PermutedDimsArray{T,N,I1}) where {T,N,I1}
    permute(size(parent(B)), static(I1))
end
function size(a::ReinterpretArray{T,N,S,A}) where {T,N,S,A}
    psize = size(parent(a))
    if _is_reshaped(typeof(a))
        if sizeof(S) === sizeof(T)
            return psize
        elseif sizeof(S) > sizeof(T)
            return (static(div(sizeof(S), sizeof(T))), psize...)
        else
            return tail(psize)
        end
    else
        return (div(first(psize) * static(sizeof(S)), static(sizeof(T))), tail(psize)...,)
    end
end
size(A::ReshapedArray) = Base.size(A)
size(A::AbstractRange) = (static_length(A),)
size(x::Base.Generator) = size(getfield(x, :iter))
size(x::Iterators.Reverse) = size(getfield(x, :itr))
size(x::Iterators.Enumerate) = size(getfield(x, :itr))
size(x::Iterators.Accumulate) = size(getfield(x, :itr))
size(x::Iterators.Pairs) = size(getfield(x, :itr))
@inline function size(x::Iterators.ProductIterator)
    eachop(_sub_size, nstatic(Val(ndims(x))), getfield(x, :iterators))
end

size(a, dim) = size(a, to_dims(a, dim))
size(a::Array, dim::Integer) = Base.arraysize(a, convert(Int, dim))
function size(a::A, dim::Integer) where {A}
    if parent_type(A) <: A
        len = known_size(A, dim)
        if len === missing
            return Int(length(axes(a, dim)))
        else
            return StaticInt(len)
        end
    else
        return size(a)[dim]
    end
end
function size(A::SubArray, dim::Integer)
    pdim = to_parent_dims(A, dim)
    if pdim > ndims(parent_type(A))
        return size(parent(A), pdim)
    else
        return static_length(A.indices[pdim])
    end
end
size(x::Iterators.Zip) = Static.reduce_tup(promote_shape, map(size, getfield(x, :is)))

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

