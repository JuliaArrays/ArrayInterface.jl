
"""
    has_dimnames(x) -> Bool

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

@generated default_dimnames(::Val{N}) where {N} = :($(ntuple(i -> Symbol(:dim_, i), N)))

"""
    dimnames(x) -> Tuple

Return the names of the dimensions for `x`.
"""
@inline dimnames(x) = dimnames(typeof(x))
function dimnames(::Type{T}) where {T}
    if parent_type(T) <: T
        return default_dimnames(Val(ndims(T)))
    else
        return dimnames(parent_type(T))
    end
end
dimnames(::Type{T}) where {T<:Transpose} = reverse(dimnames(parent_type(T)))
dimnames(::Type{T}) where {T<:Adjoint} = reverse(dimnames(parent_type(T)))
@inline function dimnames(::Type{T}) where {I1,A,T<:PermutedDimsArray{<:Any,<:Any,I1,<:Any,A}}
    ns = dimnames(A)
    return map(i -> getfield(ns, i), I1)
end
@generated function dimnames(::Type{T}) where {N,A,I,T<:SubArray{<:Any,N,A,I}}
    e = Expr(:tuple)
    d = dimnames(A)
    for i in 1:N
        if argdims(A, I.parameters[i]) > 0
            push!(e.args, d[i])
        end
    end
    return e
end

"""
    to_dims(x, d)

This returns the dimension(s) of `x` corresponding to `d`.
"""
to_dims(x, d::Integer) = Int(d)
to_dims(x, d::StaticInt) = d
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
    order_named_inds(Val(names); kw...)
    order_named_inds(Val(names), namedtuple)

Returns the tuple of index values for an array with `names`, when indexed by keywords.
Any dimensions not fixed are given as `:`, to make a slice.
An error is thrown if any keywords are used which do not occur in `nda`'s names.
"""
order_named_inds(val::Val{L}; kw...) where {L} = order_named_inds(val, kw.data)
@generated function order_named_inds(val::Val{L}, ni::NamedTuple{K}) where {L,K}
    if length(K) === 0
        return ()  # if kwargs were empty
    else
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
end

