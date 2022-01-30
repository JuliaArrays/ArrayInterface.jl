
function throw_dim_error(@nospecialize(x), @nospecialize(dim))
    throw(DimensionMismatch("$x does not have dimension corresponding to $dim"))
end

@propagate_inbounds function _promote_shape(a::Tuple{A,Vararg{Any}}, b::Tuple{B,Vararg{Any}}) where {A,B}
    (_try_static(getfield(a, 1), getfield(b, 1)), _promote_shape(tail(a), tail(b))...)
end
_promote_shape(::Tuple{}, ::Tuple{}) = ()
@propagate_inbounds function _promote_shape(::Tuple{}, b::Tuple{B}) where {B}
    (_try_static(static(1), getfield(b, 1)),)
end
@propagate_inbounds function _promote_shape(a::Tuple{A}, ::Tuple{}) where {A}
    (_try_static(static(1), getfield(a, 1)),)
end
@propagate_inbounds function Base.promote_shape(a::Tuple{Vararg{CanonicalInt}}, b::Tuple{Vararg{CanonicalInt}})
    _promote_shape(a, b)
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
    from_parent_dims(::Type{T}) -> Tuple{Vararg{Union{Int,StaticInt}}}
    from_parent_dims(::Type{T}, dim) -> Union{Int,StaticInt}

Returns the mapping from parent dimensions to child dimensions.
"""
from_parent_dims(x) = from_parent_dims(typeof(x))
from_parent_dims(::Type{T}) where {T} = nstatic(Val(ndims(T)))
from_parent_dims(::Type{T}) where {T<:VecAdjTrans} = (StaticInt(2),)
from_parent_dims(::Type{T}) where {T<:MatAdjTrans} = (StaticInt(2), One())
from_parent_dims(::Type{<:SubArray{T,N,A,I}}) where {T,N,A,I} = _from_sub_dims(I)
@generated function _from_sub_dims(::Type{I}) where {I<:Tuple}
    out = Expr(:tuple)
    dim_i = 1
    for i in 1:length(I.parameters)
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

from_parent_dims(x, dim) = from_parent_dims(typeof(x), dim)
Compat.@constprop :aggressive function from_parent_dims(::Type{T}, dim::Int)::Int where {T}
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
    to_parent_dims(::Type{T}) -> Tuple{Vararg{Union{Int,StaticInt}}}
    to_parent_dims(::Type{T}, dim) -> Union{Int,StaticInt}

Returns the mapping from child dimensions to parent dimensions.
"""
to_parent_dims(x) = to_parent_dims(typeof(x))
to_parent_dims(::Type{T}) where {T} = nstatic(Val(ndims(T)))
to_parent_dims(::Type{T}) where {T<:Union{Transpose,Adjoint}} = (StaticInt(2), One())
to_parent_dims(::Type{<:PermutedDimsArray{T,N,I}}) where {T,N,I} = static(Val(I))
to_parent_dims(::Type{<:SubArray{T,N,A,I}}) where {T,N,A,I} = _to_sub_dims(I)
@generated function _to_sub_dims(::Type{I}) where {I<:Tuple}
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

to_parent_dims(x, dim) = to_parent_dims(typeof(x), dim)
Compat.@constprop :aggressive function to_parent_dims(::Type{T}, dim::Int)::Int where {T}
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
    has_dimnames(::Type{T}) -> StaticBool

Returns `static(true)` if `x` has on or more named dimensions. If all dimensions correspond
to `static(:_)`, then `static(false)` is returned.
"""
Compat.@constprop :aggressive has_dimnames(x) = static(_is_named(known_dimnames(x)))
_is_named(x::NTuple{N,Symbol}) where {N} = x !== ntuple(Returns(:_), Val(N))
_is_named(::Any) = true

"""
    known_dimnames(::Type{T}) -> Tuple{Vararg{Union{Symbol,Missing}}}
    known_dimnames(::Type{T}, dim::Union{Int,StaticInt}) -> Union{Symbol,Missing}

Return the names of the dimensions for `x`. `:_` is used to indicate a dimension does not
have a name.
"""
@inline known_dimnames(x, dim::Integer) = _known_dimname(known_dimnames(x), canonicalize(dim))
known_dimnames(x) = known_dimnames(typeof(x))
known_dimnames(::Type{T}) where {T} = _known_dimnames(T, parent_type(T))
_known_dimnames(::Type{T}, ::Type{T}) where {T} = _unknown_dimnames(Base.IteratorSize(T))
_unknown_dimnames(::Base.HasShape{N}) where {N} = ntuple(Compat.Returns(:_), Val(N))
_unknown_dimnames(::Any) = (:_,)
function _known_dimnames(::Type{C}, ::Type{P}) where {C,P}
    eachop(_inbounds_known_dimname, to_parent_dims(C), known_dimnames(P))
end
@inline function _known_dimname(x::Tuple{Vararg{Any,N}}, dim::CanonicalInt) where {N}
    @boundscheck (dim > N || dim < 1) && return :_
    return @inbounds(x[dim])
end
@inline _inbounds_known_dimname(x, dim) = @inbounds(_known_dimname(x, dim))

"""
    dimnames(x) -> Tuple{Vararg{Union{Symbol,StaticSymbol}}}
    dimnames(x, dim::Union{Int,StaticInt}) -> Union{Symbol,StaticSymbol}

Return the names of the dimensions for `x`. `:_` is used to indicate a dimension does not
have a name.
"""
@inline dimnames(x, dim::Integer) = _dimname(dimnames(x), canonicalize(dim))
@inline dimnames(x) = _dimnames(has_parent(x), x)
@inline function _dimnames(::True, x)
    eachop(_inbounds_dimname, to_parent_dims(x), dimnames(parent(x)))
end
_dimnames(::False, x) = ntuple(_->static(:_), Val(ndims(x)))
@inline function _dimname(x::Tuple{Vararg{Any,N}}, dim::CanonicalInt) where {N}
    @boundscheck (dim > N || dim < 1) && return static(:_)
    return @inbounds(x[dim])
end
@inline _inbounds_dimname(x, dim) = @inbounds(_dimname(x, dim))

"""
    to_dims(x, dim) -> Union{Int,StaticInt}

This returns the dimension(s) of `x` corresponding to `dim`.
"""
to_dims(x, dim::Colon) = dim
to_dims(x, dim::Integer) = canonicalize(dim)
@inline function to_dims(x, dim::StaticSymbol)
    i = find_first_eq(dim, dimnames(x))
    if i === nothing
        throw_dim_error(x, dim)
    end
    return i
end
Compat.@constprop :aggressive function to_dims(x, dim::Symbol)
    i = find_first_eq(dim, map(Symbol, dimnames(x)))
    if i === nothing
        throw_dim_error(x, dim)
    end
    return i
end
to_dims(x, dims::Tuple) = map(i -> to_dims(x, i), dims)

#=
    order_named_inds(names, namedtuple)
    order_named_inds(names, subnames, inds)

Returns the tuple of index values for an array with `names`, when indexed by keywords.
Any dimensions not fixed are given as `:`, to make a slice.
An error is thrown if any keywords are used which do not occur in `nda`'s names.


1. parse into static dimnension names and key words.
2. find each dimnames in key words
3. if missing is found use Colon()
4. if (ndims - ncolon) === nkwargs then all were found, else error
=#
order_named_inds(x::Tuple, ::NamedTuple{(),Tuple{}}) = ()
function order_named_inds(x::Tuple, nd::NamedTuple{L}) where {L}
    return order_named_inds(x, static(Val(L)), Tuple(nd))
end
Compat.@constprop :aggressive function order_named_inds(
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
    return missing
end

