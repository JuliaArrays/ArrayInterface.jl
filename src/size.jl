
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
size(a::AbstractVector) = (size(a, One()),)

@inline size(x::VecAdjTrans) = (One(), static_length(parent(x)))

size(x::SubArray) = eachop(_sub_size, x.indices, to_parent_dims(x))
_sub_size(x::Tuple, ::StaticInt{dim}) where {dim} = static_length(getfield(x, dim))

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

"""
    size(A, dim)

Returns the size of `A` along dimension `dim`.
"""
size(a, dim) = size(a, to_dims(a, dim))
function size(a::A, dim::Integer) where {A}
    if parent_type(A) <: A
        len = known_size(A, dim)
        if len === nothing
            return length(axes(a, dim))::Int
        else
            return StaticInt(len)
        end
    else
        return size(parent(a), to_parent_dims(A, dim))
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
known_size(::Type{T}) where {T} = eachop(known_size, T, nstatic(Val(ndims(T))))

#=
function known_size(::Type{A}) where {T,N,P,I,A<:SubArray{T,N,P,I}}
    return eachop(_known_axis_length, I, to_parent_dims(A))
end

_known_axis_length(::Type{T}, c::StaticInt) where {T} = known_length(_get_tuple(T, c))

function known_size(::Type{R}) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    psize = known_size(A)
    if _is_reshaped(R)
        if sizeof(S) === sizeof(T)
            return psize
        elseif sizeof(S) > sizeof(T)
            return (div(sizeof(S), sizeof(T)), psize...)
        else
            return tail(psize)
        end
    else
        p1 = first(psize)
        if p1 === nothing
            return psize
        else
            return (div(p1 * sizeof(S), sizeof(T)), tail(psize)...,)
        end
    end
end
=#

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
