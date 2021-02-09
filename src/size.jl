
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

function size(B::S) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _size(size(parent(B)), B.indices, map(static_length, B.indices))
end
@generated function _size(A::Tuple{Vararg{Any,N}}, inds::I, l::L) where {N,I<:Tuple,L}
    t = Expr(:tuple)
    for n = 1:N
        if (I.parameters[n] <: Base.Slice)
            push!(t.args, :(@inbounds(_try_static(A[$n], l[$n]))))
        elseif I.parameters[n] <: Number
            nothing
        else
            push!(t.args, Expr(:ref, :l, n))
        end
    end
    Expr(:block, Expr(:meta, :inline), t)
end
@inline size(B::MatAdjTrans) = permute(size(parent(B)), to_parent_dims(B))
@inline function size(B::PermutedDimsArray{T,N,I1,I2,A}) where {T,N,I1,I2,A}
    return permute(size(parent(B)), to_parent_dims(B))
end

"""
    size(A, dim)

Returns the size of `A` along dimension `dim`.
"""
size(A::Array, dim::Integer) = Base.size(A, dim)
size(a, dim) = size(a, to_dims(a, dim))
function size(a::A, dim::Integer) where {A}
    if parent_type(A) <: A
        return _size_default(a, dim)
    else
        return size(parent(a), to_parent_dims(A, dim))
    end
end
function _size_default(a::A, dim::Integer) where {A}
    len = known_size(A, dim)
    if len === nothing
        return length(axes(a, dim))
    else
        return StaticInt(len)
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
function known_size(::Type{T}) where {T}
    return eachop(_known_axis_length, axes_types(T), nstatic(Val(ndims(T))))
end
_known_axis_length(::Type{T}, c::StaticInt) where {T} = known_length(_get_tuple(T, c))

"""
    known_size(::Type{T}, dim)

Returns the size along dimension `dim` known at compile time. If it is not known then
returns `nothing`.
"""
@inline known_size(x, dim) = known_size(typeof(x), dim)
@inline known_size(::Type{T}, dim) where {T} = known_size(T, to_dims(T, dim))
@inline function known_size(::Type{T}, dim::Integer) where {T}
    len = known_length(axes_types(T, dim))
    if len === nothing
        return nothing
    else
        return len
    end
end

###
### ReinterpretArray
###
# this handles resizing the first dimension for ReinterpretArray
resize_reinterpreted(::Nothing, ::Type{S}, ::Type{T}) where {S,T} = nothing
function resize_reinterpreted(::StaticInt{p}, ::Type{S}, ::Type{T}) where {p,S,T}
    return StaticInt(_resize_reinterpreted(p, S, T))
end
function resize_reinterpreted(p::Int, ::Type{S}, ::Type{T}) where {S,T}
    return _resize_reinterpreted(p, S, T)
end
@aggressive_constprop function _resize_reinterpreted(p::Int, ::Type{S}, ::Type{T})::Int where {S,T}
    return div(p  * sizeof(S), sizeof(T))
end

size(::ReinterpretArray{T,0}) where {T} = ()
size(x::ReinterpretArray) = _reinterpret_size(x, _is_reshaped(typeof(x)))
function _reinterpret_size(x::ReinterpretArray, ::False)
    return eachop(_size_dim, x, nstatic(Val(ndims(x))))
end
function _reinterpret_size(x::ReinterpretArray, ::True)
    if sizeof(S) === sizeof(T)
        return size(parent(x))
    elseif sizeof(S) > sizeof(T)
        return eachop(_size_dim, x, nstatic(Val(ndims(x))))
    else
        return tail(size(parent(x)))
    end
end
function _size_dim(x::ReinterpretArray{T,N,S}, dim::One) where {T,N,S}
    return resize_reinterpreted(size(parent(x), dim), S, T)
end
_size_dim(x::ReinterpretArray{T,N,S}, dim::StaticInt) where {T,N,S} = size(parent(x), dim)
function _reinterpret_known_size(::Type{A}, ::False) where {T,N,S,A<:ReinterpretArray{T,N,S}}
    return eachop(_known_size_dim, A, nstatic(Val(N)))
end
function _reinterpret_known_size(::Type{A}, ::True) where {T,N,S,A<:ReinterpretArray{T,N,S}}
    if sizeof(S) === sizeof(T)
        return known_size(parent_type(A))
    elseif sizeof(S) > sizeof(T)
        return eachop(_known_size_dim, A, nstatic(Val(N)))
    else
        return tail(known_size(parent_type(A)))
    end
end
function _known_size_dim(::Type{<:ReinterpretArray{T,N,S,A}}, dim::One) where {T,N,S,A}
    return resize_reinterpreted(known_size(A, dim), S, T)
end
function _known_size_dim(::Type{<:ReinterpretArray{T,N,S,A}}, dim::StaticInt) where {T,N,S,A}
    return known_size(A, dim)
end

