
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
@inline function size(a::A) where {A}
    if is_forwarding_wrapper(A)
        return size(parent(a))
    else
        return _maybe_size(Base.IteratorSize(A), a)
    end
end
size(a::Base.Broadcast.Broadcasted) = map(length, axes(a))

_maybe_size(::Base.HasShape{N}, a::A) where {N,A} = map(length, axes(a))
_maybe_size(::Base.HasLength, a::A) where {A} = (length(a),)

@inline size(x::SubArray) = flatten_tuples(map(Base.Fix1(_sub_size, x), sub_axes_map(typeof(x))))
@inline _sub_size(::SubArray, ::SOneTo{S}) where {S} = StaticInt(S)
_sub_size(x::SubArray, ::StaticInt{index}) where {index} = size(getfield(x.indices, index))

@inline size(B::VecAdjTrans) = (One(), length(parent(B)))
@inline function size(x::Union{PermutedDimsArray,MatAdjTrans})
    map(GetIndex{false}(size(parent(x))), to_parent_dims(x))
end
function size(a::ReinterpretArray{T,N,S,A,IsReshaped}) where {T,N,S,A,IsReshaped}
    psize = size(parent(a))
    if IsReshaped
        if sizeof(S) === sizeof(T)
            return psize
        elseif sizeof(S) > sizeof(T)
            return flatten_tuples((static(div(sizeof(S), sizeof(T))), psize))
        else
            return tail(psize)
        end
    else
        return flatten_tuples((div(first(psize) * static(sizeof(S)), static(sizeof(T))), tail(psize)))
    end
end
size(A::ReshapedArray) = Base.size(A)
size(A::AbstractRange) = (length(A),)
size(x::Base.Generator) = size(getfield(x, :iter))
size(x::Iterators.Reverse) = size(getfield(x, :itr))
size(x::Iterators.Enumerate) = size(getfield(x, :itr))
size(x::Iterators.Accumulate) = size(getfield(x, :itr))
size(x::Iterators.Pairs) = size(getfield(x, :itr))
# TODO couldn't this just be map(length, getfield(x, :iterators))
@inline function size(x::Iterators.ProductIterator)
    eachop(_sub_size, ntuple(static, StaticInt(ndims(x))), getfield(x, :iterators))
end
_sub_size(x::Tuple, ::StaticInt{dim}) where {dim} = length(getfield(x, dim))

size(a, dim) = size(a, to_dims(a, dim))
size(a::Array, dim::CanonicalInt) = Base.arraysize(a, convert(Int, dim))
function size(a::A, dim::CanonicalInt) where {A}
    if is_forwarding_wrapper(A)
        return size(parent(a), dim)
    else
        len = known_size(A, dim)
        if len === nothing
            return Int(length(axes(a, dim)))
        else
            return StaticInt(len)
        end
    end
end
size(x::Iterators.Zip) = Static.reduce_tup(promote_shape, map(size, getfield(x, :is)))

"""
    known_size(::Type{T}) -> Tuple
    known_size(::Type{T}, dim) -> Union{Int,Nothing}

Returns the size of each dimension of `A` or along dimension `dim` of `A` that is known at
compile time. If a dimension does not have a known size along a dimension then `nothing` is
returned in its position.
"""
known_size(x) = known_size(typeof(x))
@inline known_size(@nospecialize T::Type{<:Number}) = ()
@inline known_size(@nospecialize T::Type{<:VecAdjTrans}) = (1, known_length(parent_type(T)))
@inline function known_size(@nospecialize T::Type{<:Union{PermutedDimsArray,MatAdjTrans}})
    map(GetIndex{false}(known_size(parent_type(T))), to_parent_dims(T))
end
function known_size(@nospecialize T::Type{<:Diagonal})
    s = known_length(parent_type(T))
    (s, s)
end
known_size(@nospecialize T::Type{<:Union{Symmetric,Hermitian}}) = known_size(parent_type(T))
@inline function known_size(::Type{<:Base.ReinterpretArray{T,N,S,A,IsReshaped}}) where {T,N,S,A,IsReshaped}
    psize = known_size(A)
    if IsReshaped
        if sizeof(S) > sizeof(T)
            return (div(sizeof(S), sizeof(T)), psize...)
        elseif sizeof(S) < sizeof(T)
            return Base.tail(psize)
        else
            return psize
        end
    else
        if Base.issingletontype(T) || first(psize) === nothing
            return psize
        else
            return (div(first(psize) * sizeof(S), sizeof(T)), Base.tail(psize)...)
        end
    end
end

@inline function known_size(::Type{T}) where {T}
    if is_forwarding_wrapper(T)
        return known_size(parent_type(T))
    else
        return _maybe_known_size(Base.IteratorSize(T), T)
    end
end
function _maybe_known_size(::Base.HasShape{N}, ::Type{T}) where {N,T}
    eachop(_known_size, ntuple(static, StaticInt(N)), axes_types(T))
end
_maybe_known_size(::Base.IteratorSize, ::Type{T}) where {T} = (known_length(T),)
known_size(::Type{<:Base.IdentityUnitRange{I}}) where {I} = known_size(I)
known_size(::Type{<:Base.Generator{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Reverse{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Enumerate{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Accumulate{<:Any,I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Pairs{<:Any,<:Any,I}}) where {I} = known_size(I)
@inline function known_size(::Type{<:Iterators.ProductIterator{T}}) where {T}
    ntuple(i -> known_length(T.parameters[i]), Val(known_length(T)))
end
@inline function known_size(@nospecialize T::Type{<:AbstractRange})
    if is_forwarding_wrapper(T)
        return known_size(parent_type(T))
    else
        return (_range_length(known_first(T), known_step(T), known_last(T)),)
    end
end

@inline function known_size(@nospecialize T::Type{<:Union{LinearIndices,CartesianIndices}})
    I = fieldtype(T, :indices)
    ntuple(i -> known_length(I.parameters[i]), Val(ndims(T)))
end

@inline function known_size(@nospecialize T::Type{<:SubArray})
    flatten_tuples(map(Base.Fix1(_known_sub_size, T), sub_axes_map(T)))
end
_known_sub_size(@nospecialize(T::Type{<:SubArray}), ::SOneTo{S}) where {S} = S
function _known_sub_size(@nospecialize(T::Type{<:SubArray}), ::StaticInt{index}) where {index}
    known_size(fieldtype(fieldtype(T, :indices), index))
end

# 1. `Zip` doesn't check that its collections are compatible (same size) at construction,
#   but we assume as much b/c otherwise it will error while iterating. So we promote to the
#   known size if matching a `Nothing` and `Int` size.
# 2. `promote_shape(::Tuple{Vararg{CanonicalInt}}, ::Tuple{Vararg{CanonicalInt}})` promotes
#   trailing dimensions (which must be of size 1), to `static(1)`. We want to stick to
#   `Nothing` and `Int` types, so we do one last pass to ensure everything is dynamic
@inline function known_size(::Type{<:Iterators.Zip{T}}) where {T}
    dynamic(reduce_tup(Static._promote_shape, eachop(_unzip_size, ntuple(static, StaticInt(known_length(T))), T)))
end
_unzip_size(::Type{T}, n::StaticInt{N}) where {T,N} = known_size(field_type(T, n))
_known_size(::Type{T}, dim::StaticInt) where {T} = known_length(field_type(T, dim))
@inline known_size(x, dim) = known_size(typeof(x), dim)
@inline known_size(::Type{T}, dim) where {T} = known_size(T, to_dims(T, dim))
known_size(T::Type, dim::CanonicalInt) = ndims(T) < dim ? 1 : known_size(T)[dim]

"""
    length(A) -> Union{Int,StaticInt}

Returns the length of `A`.  If the length is known at compile time, it is
returned as `Static` number.  Otherwise, `ArrayInterface.length(A)` is identical
to `Base.length(A)`.

```julia
julia> using StaticArrays, ArrayInterface

julia> A = @SMatrix rand(3,4);

julia> ArrayInterface.length(A)
static(12)
```
"""
@inline length(a::UnitRange{T}) where {T} = last(a) - first(a) + oneunit(T)
@inline length(x) = Static.maybe_static(known_length, Base.length, x)

# Alias to to-be-depreciated internal function
const static_length = length

"""
    known_length(::Type{T}) -> Union{Int,Nothing}

If `length` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.
"""
known_length(x) = known_length(typeof(x))
known_length(::Type{<:NamedTuple{L}}) where {L} = length(L)
known_length(::Type{T}) where {T<:Slice} = known_length(parent_type(T))
known_length(::Type{<:Tuple{Vararg{Any,N}}}) where {N} = N
known_length(::Type{<:Number}) = 1
known_length(::Type{<:AbstractCartesianIndex{N}}) where {N} = N
known_length(::Type{T}) where {T} = _maybe_known_length(Base.IteratorSize(T), T)
function known_length(::Type{<:Iterators.Flatten{I}}) where {I}
  _prod_or_nothing((known_length(I),known_length(eltype(I))))
end

_prod_or_nothing(x::Tuple{Vararg{Int}}) = prod(x)
_prod_or_nothing(_) = nothing

_maybe_known_length(::Base.HasShape, ::Type{T}) where {T} = _prod_or_nothing(known_size(T))
_maybe_known_length(::Base.IteratorSize, ::Type) = nothing
