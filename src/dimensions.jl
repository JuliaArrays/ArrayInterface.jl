
#julia> @btime ArrayInterface.is_increasing(ArrayInterface.nstatic(Val(10)))
#  0.045 ns (0 allocations: 0 bytes)
#ArrayInterface.True()
function is_increasing(perm::Tuple{StaticInt{X},StaticInt{Y},Vararg}) where {X, Y}
    if X <= Y
        return is_increasing(tail(perm))
    else
        return False()
    end
end
function is_increasing(perm::Tuple{StaticInt{X},StaticInt{Y}}) where {X, Y}
    if X <= Y
        return True()
    else
        return False()
    end
end
is_increasing(::Tuple{StaticInt{X}}) where {X} = True()

"""
    from_parent_dims(::Type{T}) -> Bool

Returns the mapping from parent dimensions to child dimensions.
"""
from_parent_dims(::Type{T}) where {T} = nstatic(Val(ndims(T)))
from_parent_dims(::Type{T}) where {T<:Union{Transpose,Adjoint}} = (StaticInt(2), One())
from_parent_dims(::Type{<:SubArray{T,N,A,I}}) where {T,N,A,I} = _from_sub_dims(A, I)
@generated function _from_sub_dims(::Type{A}, ::Type{I}) where {A,N,I<:Tuple{Vararg{Any,N}}}
    out = Expr(:tuple)
    n = 1
    for p in I.parameters
        if argdims(A, p) > 0
            push!(out.args, :(StaticInt($n)))
            n += 1
        else
            push!(out.args, :(StaticInt(0)))
        end
    end
    out
end
function from_parent_dims(::Type{<:PermutedDimsArray{T,N,<:Any,I}}) where {T,N,I}
    return _val_to_static(Val(I))
end

"""
    to_parent_dims(::Type{T}) -> Bool

Returns the mapping from child dimensions to parent dimensions.
"""
to_parent_dims(x) = to_parent_dims(typeof(x))
to_parent_dims(::Type{T}) where {T} = nstatic(Val(ndims(T)))
to_parent_dims(::Type{T}) where {T<:Union{Transpose,Adjoint}} = (StaticInt(2), One())
to_parent_dims(::Type{<:PermutedDimsArray{T,N,I}}) where {T,N,I} = _val_to_static(Val(I))
to_parent_dims(::Type{<:SubArray{T,N,A,I}}) where {T,N,A,I} = _to_sub_dims(A, I)
@generated function _to_sub_dims(::Type{A}, ::Type{I}) where {A,N,I<:Tuple{Vararg{Any,N}}}
    out = Expr(:tuple)
    n = 1
    for p in I.parameters
        if argdims(A, p) > 0
            push!(out.args, :(StaticInt($n)))
        end
        n += 1
    end
    out
end

"""
    has_dimnames(::Type{T}) -> Bool

Returns `true` if `x` has names for each dimension.
"""
@inline has_dimnames(x) = has_dimnames(typeof(x))
function has_dimnames(::Type{T}) where {T}
    if parent_type(T) <: T
        return false
    else
        return has_dimnames(parent_type(T))
    end
end

"""
    dimnames(::Type{T}) -> Tuple{Vararg{Symbol}}
    dimnames(::Type{T}, d) -> Symbol

Return the names of the dimensions for `x`.
"""
@inline dimnames(x) = dimnames(typeof(x))
@inline dimnames(x, i::Integer) = dimnames(typeof(x), i)
@inline dimnames(::Type{T}, d::Integer) where {T} = getfield(dimnames(T), Int(d))
@inline function dimnames(::Type{T}) where {T}
    if parent_type(T) <: T
        return ntuple(i -> :_, Val(ndims(T)))
    else
        return dimnames(parent_type(T))
    end
end
@inline function dimnames(::Type{T}) where {T<:Union{Transpose,Adjoint}}
    return _transpose_dimnames(Val(dimnames(parent_type(T))))
end
# inserting the Val here seems to help inferability; I got a test failure without it.
function _transpose_dimnames(::Val{S}) where {S}
    if length(S) == 1
        (:_, first(S))
    elseif length(S) == 2
        (last(S), first(S))
    else
        throw("Can't transpose $S of dim $(length(S)).")
    end
end
@inline _transpose_dimnames(x::Tuple{Symbol,Symbol}) = (last(x), first(x))
@inline _transpose_dimnames(x::Tuple{Symbol}) = (:_, first(x))

@inline function dimnames(::Type{T}) where {I,T<:PermutedDimsArray{<:Any,<:Any,I}}
    return map(i -> dimnames(parent_type(T), i), I)
end
function dimnames(::Type{T}) where {P,I,T<:SubArray{<:Any,<:Any,P,I}}
    return _sub_array_dimnames(Val(dimnames(P)), Val(argdims(P, I)))
end
@generated function _sub_array_dimnames(::Val{L}, ::Val{I}) where {L,I}
    e = Expr(:tuple)
    nl = length(L)
    for i in 1:length(I)
        if I[i] > 0
            if nl < i
                push!(e.args, QuoteNode(:_))
            else
                push!(e.args, QuoteNode(L[i]))
            end
        end
    end
    return e
end

_int_or_sint(x::Integer) = Int(x)
_int_or_sint(x::StaticInt) = x

"""
    to_dims(x[, d])

This returns the dimension(s) of `x` corresponding to `d`.
"""
to_dims(x, d) = to_dims(dimnames(x), d)
to_dims(x, d::Integer) = _int_or_sint(d)
to_dims(x::Tuple{Vararg{Symbol}}, d::Integer) = _int_or_sint(d)
to_dims(x::Tuple{Vararg{Symbol}}, d::Colon) = d   # `:` is the default for most methods that take `dims`
@inline to_dims(x::Tuple{Vararg{Symbol}}, d::Tuple) = map(i -> to_dims(x, i), d)
@inline function to_dims(x::Tuple{Vararg{Symbol}}, d::Symbol)::Int
    i = _sym_to_dim(x, d)
    if i === 0
        throw(ArgumentError("Specified name ($(repr(d))) does not match any dimension name ($(x))"))
    end
    return i
end
Base.@pure function _sym_to_dim(x::Tuple{Vararg{Symbol,N}}, sym::Symbol) where {N}
    for i in 1:N
        getfield(x, i) === sym && return i
    end
    return 0
end

"""
    tuple_issubset

A version of `issubset` sepecifically for `Tuple`s of `Symbol`s, that is `@pure`.
This helps it get optimised out of existance. It is less of an abuse of `@pure` than
most of the stuff for making `NamedTuples` work.
"""
Base.@pure function tuple_issubset(
    lhs::Tuple{Vararg{Symbol,N}}, rhs::Tuple{Vararg{Symbol,M}}
) where {N,M}
    N <= M || return false
    for a in lhs
        found = false
        for b in rhs
            found |= a === b
        end
        found || return false
    end
    return true
end

"""
    order_named_inds(Val(names); kwargs...)
    order_named_inds(Val(names), namedtuple)

Returns the tuple of index values for an array with `names`, when indexed by keywords.
Any dimensions not fixed are given as `:`, to make a slice.
An error is thrown if any keywords are used which do not occur in `nda`'s names.
"""
@inline function order_named_inds(val::Val{L}; kwargs...) where {L}
    if isempty(kwargs)
        return ()
    else
        return order_named_inds(val, kwargs.data)
    end
end
@generated function order_named_inds(val::Val{L}, ni::NamedTuple{K}) where {L,K}
    tuple_issubset(K, L) || throw(DimensionMismatch("Expected subset of $L, got $K"))
    exs = map(L) do n
        if Base.sym_in(n, K)
            qn = QuoteNode(n)
            :(getfield(ni, $qn))
        else
            :(Colon())
        end
    end
    return Expr(:tuple, exs...)
end
@generated function _perm_tuple(::Type{T}, ::Val{P}) where {T,P}
    out = Expr(:curly, :Tuple)
    for p in P
        push!(out.args, T.parameters[p])
    end
    Expr(:block, Expr(:meta, :inline), out)
end

"""
    axes_types(::Type{T}[, d]) -> Type

Returns the type of the axes for `T`
"""
axes_types(x) = axes_types(typeof(x))
axes_types(x, d) = axes_types(typeof(x), d)
@inline axes_types(::Type{T}, d) where {T} = axes_types(T).parameters[to_dims(T, d)]
function axes_types(::Type{T}) where {T}
    if parent_type(T) <: T
        return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},ndims(T)}}
    else
        return axes_types(parent_type(T))
    end
end
function axes_types(::Type{T}) where {T<:Adjoint}
    return _perm_tuple(axes_types(parent_type(T)), Val((2, 1)))
end
function axes_types(::Type{T}) where {T<:Transpose}
    return _perm_tuple(axes_types(parent_type(T)), Val((2, 1)))
end
function axes_types(::Type{T}) where {I1,T<:PermutedDimsArray{<:Any,<:Any,I1}}
    return _perm_tuple(axes_types(parent_type(T)), Val(I1))
end
function axes_types(::Type{T}) where {T<:AbstractRange}
    if known_length(T) === nothing
        return Tuple{OptionallyStaticUnitRange{One,Int}}
    else
        return Tuple{OptionallyStaticUnitRange{One,StaticInt{known_length(T)}}}
    end
end

@inline function axes_types(::Type{T}) where {P,I,T<:SubArray{<:Any,<:Any,P,I}}
    return _sub_axes_types(Val(ArrayStyle(T)), I, axes_types(P))
end
@inline function axes_types(::Type{T}) where {T<:Base.ReinterpretArray}
    return _reinterpret_axes_types(
        axes_types(parent_type(T)),
        eltype(T),
        eltype(parent_type(T)),
    )
end
function axes_types(::Type{T}) where {N,T<:Base.ReshapedArray{<:Any,N}}
    return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},N}}
end

# These methods help handle identifying axes that don't directly propagate from the
# parent array axes. They may be worth making a formal part of the API, as they provide
# a low traffic spot to change what axes_types produces.
@inline function sub_axis_type(::Type{A}, ::Type{I}) where {A,I}
    if known_length(I) === nothing
        return OptionallyStaticUnitRange{One,Int}
    else
        return OptionallyStaticUnitRange{One,StaticInt{known_length(I)}}
    end
end
@generated function _sub_axes_types(
    ::Val{S},
    ::Type{I},
    ::Type{PI},
) where {S,I<:Tuple,PI<:Tuple}
    out = Expr(:curly, :Tuple)
    d = 1
    for i in I.parameters
        ad = argdims(S, i)
        if ad > 0
            push!(out.args, :(sub_axis_type($(PI.parameters[d]), $i)))
            d += ad
        else
            d += 1
        end
    end
    Expr(:block, Expr(:meta, :inline), out)
end
@inline function reinterpret_axis_type(::Type{A}, ::Type{T}, ::Type{S}) where {A,T,S}
    if known_length(A) === nothing
        return OptionallyStaticUnitRange{One,Int}
    else
        return OptionallyStaticUnitRange{
            One,
            StaticInt{Int(known_length(A) / (sizeof(T) / sizeof(S)))},
        }
    end
end
@generated function _reinterpret_axes_types(
    ::Type{I},
    ::Type{T},
    ::Type{S},
) where {I<:Tuple,T,S}
    out = Expr(:curly, :Tuple)
    for i = 1:length(I.parameters)
        if i === 1
            push!(out.args, reinterpret_axis_type(I.parameters[1], T, S))
        else
            push!(out.args, I.parameters[i])
        end
    end
    Expr(:block, Expr(:meta, :inline), out)
end

"""
    known_size(::Type{T}[, d]) -> Tuple

Returns the size of each dimension for `T` known at compile time. If a dimension does not
have a known size along a dimension then `nothing` is returned in its position.
"""
@inline known_size(x, d) = known_size(x)[to_dims(x, d)]
known_size(x) = known_size(typeof(x))
function known_size(::Type{T}) where {T}
    return eachop(_known_axis_length, axes_types(T), nstatic(Val(ndims(T))))
end
_known_axis_length(::Type{T}, c::StaticInt) where {T} = known_length(_get_tuple(T, c))

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
@inline function size(a::A) where {A}
    if parent_type(A) <: A
        return Base.size(a)  # TODO should this throw an error instead?
    else
        return size(parent(a))
    end
end
@inline function size(a::A, d) where {A}
    if parent_type(A) <: A
        return Base.size(a, to_dims(A, d))
    else
        return size(parent(a), to_dims(A, d)) 
    end
end
@inline function size(x::LinearAlgebra.Adjoint{T,V}) where {T,V<:AbstractVector{T}}
    return (One(), static_length(x))
end
@inline function size(x::LinearAlgebra.Transpose{T,V}) where {T,V<:AbstractVector{T}}
    return (One(), static_length(x))
end

function size(B::S) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _size(size(parent(B)), B.indices, map(static_length, B.indices))
end
function strides(B::S) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _strides(strides(parent(B)), B.indices)
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
@inline size(v::AbstractVector) = (static_length(v),)
@inline size(B::MatAdjTrans) = permute(size(parent(B)), Val{(2, 1)}())
@inline function size(B::PermutedDimsArray{T,N,I1,I2,A}) where {T,N,I1,I2,A}
    return permute(size(parent(B)), Val{I1}())
end
@inline size(A::AbstractArray, ::StaticInt{N}) where {N} = size(A)[N]
@inline size(A::AbstractArray, ::Val{N}) where {N} = size(A)[N]

"""
    axes(A, d)

Return a valid range that maps to each index along dimension `d` of `A`.
"""
@inline function axes(a::A, d) where {A}
    if parent_type(A) <: A
        return Base.axes(a, Int(to_dims(A, d)))
    else
        return axes(parent(a), to_dims(A, d))
    end
end
axes(a::PermutedDimsArray, d) = Base.axes(a, Int(to_dims(a, d)))
axes(a::SubArray, d) = Base.axes(a, Int(to_dims(a, d)))
axes(a::Union{Adjoint,Transpose}, d) = Base.axes(a, Int(to_dims(a, d)))

"""
    axes(A)

Return a tuple of ranges where each range maps to each element along a dimension of `A`.
"""
@inline function axes(a::A) where {A}
    if parent_type(A) <: A
        return Base.axes(a)
    else
        return axes(parent(a))
    end
end
axes(a::PermutedDimsArray) = Base.axes(a)
axes(a::SubArray) = Base.axes(a)
axes(a::Union{Adjoint,Transpose}) = Base.axes(a)

