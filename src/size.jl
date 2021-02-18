
"""
  size(A)

Returns the size of `A`. If the size of any axes are known at compile time,
these should be returned as `Static` numbers. For example:
```julia
julia> using StaticArrays, ArrayInterface

julia> A = @SMatrix rand(3,4);

julia> ArrayInterface.size(A)
(StaticInt{3}(), StaticInt{4}())
```
"""
function size(a::A) where {A}
    if parent_type(A) <: A
        return map(static_length, axes(a))
    else
        return size(parent(a))
    end
end
#size(a::AbstractVector) = (size(a, One()),)

size(x::SubArray) = eachop(_sub_size, x.indices; iterator=to_parent_dims(x))
_sub_size(x::Tuple, ::StaticInt{dim}) where {dim} = static_length(getfield(x, dim))

@inline size(B::VecAdjTrans) = (One(), length(parent(B)))
@inline size(B::MatAdjTrans) = permute(size(parent(B)), to_parent_dims(B))
@inline function size(B::PermutedDimsArray{T,N,I1,I2,A}) where {T,N,I1,I2,A}
    return permute(size(parent(B)), to_parent_dims(B))
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
size(A::ReshapedArray) = A.dims
size(A::AbstractRange) = (static_length(A),)

"""
    size(A, dim)

Returns the size of `A` along dimension `dim`.
"""
size(a, dim) = size(a, to_dims(a, dim))
function size(a::A, dim::Integer) where {A}
    if parent_type(A) <: A
        len = known_size(A, dim)
        if len === nothing
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

"""
    known_size(::Type{T}) -> Tuple

Returns the size of each dimension for `T` known at compile time. If a dimension does not
have a known size along a dimension then `nothing` is returned in its position.
"""
known_size(x) = known_size(typeof(x))
known_size(::Type{T}) where {T} = eachop(known_size, T; iterator=nstatic(Val(ndims(T))))

"""
    known_size(::Type{T}, dim)

Returns the size along dimension `dim` known at compile time. If it is not known then
returns `nothing`.
"""
@inline known_size(x, dim) = known_size(typeof(x), dim)
@inline known_size(::Type{T}, dim) where {T} = known_size(T, to_dims(T, dim))
@inline function known_size(::Type{T}, dim::Integer) where {T}
    if ndims(T) < dim
        return 1
    else
        return known_length(axes_types(T, dim))
    end
end

