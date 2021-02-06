
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
from_parent_dims(::Type{<:PermutedDimsArray{T,N,<:Any,I}}) where {T,N,I} = static(Val(I))

"""
    to_parent_dims(::Type{T}) -> Bool

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

# this takes the place of dimension names that aren't defined
const SUnderscore = StaticSymbol(:_)

"""
    dimnames(::Type{T}) -> Tuple{Vararg{StaticSymbol}}
    dimnames(::Type{T}, dim) -> StaticSymbol

Return the names of the dimensions for `x`.
"""
@inline dimnames(x) = dimnames(typeof(x))
@inline dimnames(x, dim::Int) = dimnames(typeof(x), dim)
@inline dimnames(x, dim::StaticInt) = dimnames(typeof(x), dim)
@inline function dimnames(::Type{T}, ::StaticInt{dim}) where {T,dim}
    if ndims(T) < dim
        return SUnderscore
    else
        return getfield(dimnames(T), dim)
    end
end
@inline function dimnames(::Type{T}, dim::Int) where {T}
    if ndims(T) < dim
        return SUnderscore
    else
        return getfield(dimnames(T), dim)
    end
end
@inline function dimnames(::Type{T}) where {T}
    if parent_type(T) <: T
        return ntuple(_ -> SUnderscore, Val(ndims(T)))
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
        (SUnderscore, first(S))
    elseif length(S) == 2
        (last(S), first(S))
    else
        throw("Can't transpose $S of dim $(length(S)).")
    end
end
@inline _transpose_dimnames(x::Tuple{Symbol,Symbol}) = (last(x), first(x))
@inline _transpose_dimnames(x::Tuple{Symbol}) = (SUnderscore, first(x))

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
                push!(e.args, :(ArrayInterface.SUnderscore))
            else
                push!(e.args, QuoteNode(L[i]))
            end
        end
    end
    return e
end

_to_int(x::Integer) = Int(x)
_to_int(x::StaticInt) = x

function no_dimname_error(@nospecialize(x), @nospecialize(dim))
    throw(ArgumentError("($(repr(dim))) does not correspond to any dimension of ($(x))"))
end

"""
    to_dims(::Type{T}, dim) -> Integer

This returns the dimension(s) of `x` corresponding to `d`.
"""
to_dims(x, dim) = to_dims(typeof(x), dim)
to_dims(::Type{T}, dim::Integer) where {T} = _to_int(dim)
to_dims(::Type{T}, dim::Colon) where {T} = dim
function to_dims(::Type{T}, dim::StaticSymbol) where {T}
    i = find_first_eq(dim, dimnames(T))
    i === nothing && no_dimname_error(T, dim)
    return i
end
@inline function to_dims(::Type{T}, dim::Symbol) where {T}
    i = find_first_eq(dim, Symbol.(dimnames(T)))
    i === nothing && no_dimname_error(T, dim)
    return i
end
#=
    return i
    i = 1
    out = 0
    for s in dimnames(T)
        if Symbol(s) === dim
            out = i
            break
        else
            i += i
        end
    end
    if out === 0
        no_dimname_error(T, dim)
    end
    return out
end
=#

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
function order_named_inds(x::Tuple{Vararg{Any,N}}, nd::Tuple, inds::Tuple) where {N}
    out = eachop(((x, nd, inds), i) -> order_named_inds(x, nd, inds, i), (x, nd, inds), nstatic(Val(N)))
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


#=
name_to_idx(name::StaticSymbol, kwargs::Tuple, inds::Tuple, )
name_to_idx(name::StaticSymbol, kwargs::Tuple, inds::Tuple) = _name_to_index(find_first_eq(), inds)
_name_to_index(::Zero, ::Tuple) = Colon()
_name_to_index(::StaticInt{N}, inds::Tuple) where {N} = getfield(inds, N)

#    return permute(inds, static_find_all_in(nd, x))
_colon_or_inds(inds::Tuple, ::Zero) = :
_colon_or_inds(inds::Tuple, ::StaticInt{I}) where {I} = getfield(inds, I)

n_i -> _colon_or_inds(inds, find_first_eq(n_i, x))
# FIXME this needs to insert a colon on missing names

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
=#

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
@inline size(A) = Base.size(A)
@inline size(A, d::Integer) = size(A)[Int(d)]
@inline size(A, d) = Base.size(A, to_dims(A, d))
@inline size(x::VecAdjTrans) = (One(), static_length(x))

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
@inline axes(A, d) = axes(A, to_dims(A, d))
@inline axes(A, d::Integer) = axes(A)[Int(d)]

"""
    axes(A)

Return a tuple of ranges where each range maps to each element along a dimension of `A`.
"""
@inline axes(A) = Base.axes(A)

