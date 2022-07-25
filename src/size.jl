
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
