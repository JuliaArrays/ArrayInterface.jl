
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
@inline dimnames(::Type{T}, d::Integer) where {T} = getfield(dimnames(T), to_dims(T, d))
@inline function dimnames(::Type{T}) where {T}
    if parent_type(T) <: T
        return ntuple(i -> :_, Val(ndims(T)))
    else
        return dimnames(parent_type(T))
    end
end
@inline function dimnames(::Type{T}) where {T<:Union{Transpose,Adjoint}}
    return _transpose_dimnames(dimnames(parent_type(T)))
end
_transpose_dimnames(x::Tuple{Symbol,Symbol}) = (last(x), first(x))
_transpose_dimnames(x::Tuple{Symbol}) = (:_, first(x))

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

"""
    to_dims(x[, d])

This returns the dimension(s) of `x` corresponding to `d`.
"""
to_dims(x, d::Integer) = Int(d)
to_dims(x, d::Colon) = d   # `:` is the default for most methods that take `dims`
@inline to_dims(x, d::Tuple) = map(i -> to_dims(x, i), d)
@inline function to_dims(x, d::Symbol)::Int
    i = _sym_to_dim(dimnames(x), d)
    if i === 0
        throw(ArgumentError("Specified name ($(repr(d))) does not match any dimension name ($(dimnames(x)))"))
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
    lhs::Tuple{Vararg{Symbol,N}}, rhs::Tuple{Vararg{Symbol,M}},
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
size(A) = Base.size(A)
size(A, d) = Base.size(A, to_dims(A, d))

"""
    axes(A, d)

Return a valid range that maps to each index along dimension `d` of `A`.
"""
axes(A, d) = Base.axes(A, to_dims(A, d))

"""
    axes(A)

Return a tuple of ranges where each range maps to each element along a dimension of `A`.
"""
axes(A) = Base.axes(A)

