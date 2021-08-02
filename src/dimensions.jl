
function throw_dim_error(@nospecialize(x), @nospecialize(dim))
    throw(DimensionMismatch("$x does not have dimension corresponding to $dim"))
end

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
    from_parent_dims(::Type{T}) -> Tuple

Returns the mapping from parent dimensions to child dimensions.
"""
from_parent_dims(x) = from_parent_dims(typeof(x))
from_parent_dims(::Type{T}) where {T} = nstatic(Val(ndims(T)))
from_parent_dims(::Type{T}) where {T<:VecAdjTrans} = (StaticInt(2),)
from_parent_dims(::Type{T}) where {T<:MatAdjTrans} = (StaticInt(2), One())
from_parent_dims(::Type{<:SubArray{T,N,A,I}}) where {T,N,A,I} = _from_sub_dims(A, I)
@generated function _from_sub_dims(::Type{A}, ::Type{I}) where {A,I<:Tuple}
    out = Expr(:tuple)
    dim_i = 1
    for i in 1:ndims(A)
        p = I.parameters[i]
        if p <: Integer
            push!(out.args, :(StaticInt(0)))
        else
            push!(out.args, :(StaticInt($dim_i)))
            dim_i += 1
        end
    end
    out
end
from_parent_dims(::Type{<:PermutedDimsArray{T,N,<:Any,I}}) where {T,N,I} = static(Val(I))

function from_parent_dims(::Type{R}) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    if !_is_reshaped(R) || sizeof(S) === sizeof(T)
        return nstatic(Val(ndims(A)))
    elseif sizeof(S) > sizeof(T)
        return tail(nstatic(Val(ndims(A) + 1)))
    else  # sizeof(S) < sizeof(T)
        return (Zero(), nstatic(Val(N))...)
    end
end

"""
    from_parent_dims(::Type{T}, dim) -> Integer

Returns the mapping from child dimensions to parent dimensions.
"""
from_parent_dims(x, dim) = from_parent_dims(typeof(x), dim)
@aggressive_constprop function from_parent_dims(::Type{T}, dim::Int)::Int where {T}
    if dim > ndims(T)
        return static(ndims(parent_type(T)) + dim - ndims(T))
    elseif dim > 0
        return @inbounds(getfield(from_parent_dims(T), dim))
    else
        throw_dim_error(T, dim)
    end
end

function from_parent_dims(::Type{T}, ::StaticInt{dim}) where {T,dim}
    if dim > ndims(T)
        return static(ndims(parent_type(T)) + dim - ndims(T))
    elseif dim > 0
        return @inbounds(getfield(from_parent_dims(T), dim))
    else
        throw_dim_error(T, dim)
    end
end

"""
    to_parent_dims(::Type{T}) -> Tuple

Returns the mapping from child dimensions to parent dimensions.
"""
to_parent_dims(x) = to_parent_dims(typeof(x))
to_parent_dims(::Type{T}) where {T} = nstatic(Val(ndims(T)))
to_parent_dims(::Type{T}) where {T<:Union{Transpose,Adjoint}} = (StaticInt(2), One())
to_parent_dims(::Type{<:PermutedDimsArray{T,N,I}}) where {T,N,I} = static(Val(I))
to_parent_dims(::Type{<:SubArray{T,N,A,I}}) where {T,N,A,I} = _to_sub_dims(A, I)
@generated function _to_sub_dims(::Type{A}, ::Type{I}) where {A,N,I<:Tuple{Vararg{Any,N}}}
    out = Expr(:tuple)
    n = 1
    for p in I.parameters
        if !(p <: Integer)
            push!(out.args, :(StaticInt($n)))
        end
        n += 1
    end
    out
end
function to_parent_dims(::Type{R}) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    pdims = nstatic(Val(ndims(A)))
    if !_is_reshaped(R) || sizeof(S) === sizeof(T)
        return pdims
    elseif sizeof(S) > sizeof(T)
        return (Zero(), pdims...,)
    else
        return tail(pdims)
    end
end

"""
    to_parent_dims(::Type{T}, dim) -> Integer

Returns the mapping from child dimensions to parent dimensions.
"""
to_parent_dims(x, dim) = to_parent_dims(typeof(x), dim)
@aggressive_constprop function to_parent_dims(::Type{T}, dim::Int)::Int where {T}
    if dim > ndims(T)
        return static(ndims(parent_type(T)) + dim - ndims(T))
    elseif dim > 0
        return @inbounds(getfield(to_parent_dims(T), dim))
    else
        throw_dim_error(T, dim)
    end
end

function to_parent_dims(::Type{T}, ::StaticInt{dim}) where {T,dim}
    if dim > ndims(T)
        return static(ndims(parent_type(T)) + dim - ndims(T))
    elseif dim > 0
        return @inbounds(getfield(to_parent_dims(T), dim))
    else
        throw_dim_error(T, dim)
    end
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

# this takes the place of dimension names that aren't defined
const SUnderscore = StaticSymbol(:_)

"""
    dimnames(::Type{T}) -> Tuple{Vararg{StaticSymbol}}
    dimnames(::Type{T}, dim) -> StaticSymbol

Return the names of the dimensions for `x`.
"""
@inline dimnames(x) = dimnames(typeof(x))
@inline dimnames(x, dim) = dimnames(typeof(x), dim)
@inline function dimnames(::Type{T}, dim) where {T}
    if parent_type(T) <: T
        return SUnderscore
    else
        return dimnames(parent_type(T), to_parent_dims(T, dim))
    end
end
@inline function dimnames(::Type{T}) where {T}
    if parent_type(T) <: T
        return ntuple(_ -> SUnderscore, Val(ndims(T)))
    else
        perm = to_parent_dims(T)
        if invariant_permutation(perm, perm) isa True
            return dimnames(parent_type(T))
        else
            return eachop(dimnames, perm, parent_type(T))
        end
    end
end
function dimnames(::Type{T}) where {T<:SubArray}
    return eachop(dimnames, to_parent_dims(T), parent_type(T))
end

"""
    to_dims(::Type{T}, dim) -> Integer

This returns the dimension(s) of `x` corresponding to `d`.
"""
to_dims(x, dim) = to_dims(typeof(x), dim)
to_dims(::Type{T}, dim::Integer) where {T} = canonicalize(dim)
to_dims(::Type{T}, dim::Colon) where {T} = dim
function to_dims(::Type{T}, dim::StaticSymbol) where {T}
    i = find_first_eq(dim, dimnames(T))
    if i === nothing
        throw_dim_error(T, dim)
    end
    return i
end
@aggressive_constprop function to_dims(::Type{T}, dim::Symbol) where {T}
    i = find_first_eq(dim, map(Symbol, dimnames(T)))
    if i === nothing
        throw_dim_error(T, dim)
    end
    return i
end
to_dims(::Type{T}, dims::Tuple) where {T} = map(i -> to_dims(T, i), dims)

#=
    order_named_inds(names, namedtuple)
    order_named_inds(names, subnames, inds)

Returns the tuple of index values for an array with `names`, when indexed by keywords.
Any dimensions not fixed are given as `:`, to make a slice.
An error is thrown if any keywords are used which do not occur in `nda`'s names.


1. parse into static dimnension names and key words.
2. find each dimnames in key words
3. if nothing is found use Colon()
4. if (ndims - ncolon) === nkwargs then all were found, else error
=#
order_named_inds(x::Tuple, ::NamedTuple{(),Tuple{}}) = ()
function order_named_inds(x::Tuple, nd::NamedTuple{L}) where {L}
    return order_named_inds(x, static(Val(L)), Tuple(nd))
end
@aggressive_constprop function order_named_inds(
    x::Tuple{Vararg{Any,N}},
    nd::Tuple,
    inds::Tuple
) where {N}

    out = eachop(order_named_inds, nstatic(Val(N)), x, nd, inds)
    _order_named_inds_check(out, length(nd))
    return out
end
function order_named_inds(x::Tuple, nd::Tuple, inds::Tuple, ::StaticInt{dim}) where {dim}
    index = find_first_eq(getfield(x, dim), nd)
    if index === nothing
        return Colon()
    else
        return @inbounds(inds[index])
    end
end

ncolon(x::Tuple{Colon,Vararg}, n::Int) = ncolon(tail(x), n + 1)
ncolon(x::Tuple{Any,Vararg}, n::Int) = ncolon(tail(x), n)
ncolon(x::Tuple{Colon}, n::Int) = n + 1
ncolon(x::Tuple{Any}, n::Int) = n
function _order_named_inds_check(inds::Tuple{Vararg{Any,N}}, nkwargs::Int) where {N}
    if (N - ncolon(inds, 0)) !== nkwargs
        error("Not all keywords matched dimension names.")
    end
    return nothing
end

