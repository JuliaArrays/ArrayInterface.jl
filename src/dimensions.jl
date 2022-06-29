

_init_dimsmap(x) = _init_dimsmap(IndicesInfo(x))
function _init_dimsmap(::IndicesInfo{N,pdims,cdims}) where {N,pdims,cdims}
    ntuple(i -> static(getfield(pdims, i)), length(pdims)),
    ntuple(i -> static(getfield(cdims, i)), length(pdims))
end
# TODO move these
Static.static(::Colon) = (:)
Static.static(::Nothing) = nothing

"""
    to_parent_dims(::Type{T}) -> Tuple{Vararg{Union{StaticInt,Tuple{Vararg{StaticInt}}}}}

Returns the mapping from child dimensions to parent dimensions.

!!! Warning
    This method is still experimental and may change without notice.

"""
to_parent_dims(@nospecialize x) = to_parent_dims(typeof(x))
@inline function to_parent_dims(@nospecialize T::Type{<:SubArray})
    to_parent_dims(IndicesInfo{ndims(parent_type(T))}(fieldtype(T, :indices)))
end
function to_parent_dims(::IndicesInfo{N,pdims,cdims}) where {N,pdims,cdims}
    flatten_tuples(ntuple(length(cdims)) do i
        cdim_i = getfield(cdims, i)
        if cdim_i isa Tuple
            pdim_i = static(getfield(pdims, i))
            ntuple(Compat.Returns(pdim_i), length(cdim_i))
        else
            cdim_i === 0 ? () : (static(getfield(pdims, i)),)
        end
    end)
end
to_parent_dims(@nospecialize T::Type{<:MatAdjTrans}) = (StaticInt(2), StaticInt(1))
to_parent_dims(@nospecialize T::Type{<:PermutedDimsArray}) = getfield(_permdims(T), 1)

function _permdims(::Type{<:PermutedDimsArray{<:Any,<:Any,I1,I2}}) where {I1,I2}
    (map(static, I1), map(static, I2))
end

function throw_dim_error(@nospecialize(x), @nospecialize(dim))
    throw(DimensionMismatch("$x does not have dimension corresponding to $dim"))
end

# Base will sometomes demote statically known slices in `SubArray` to `OneTo{Int}` so we
# provide the parent mapping to check for static size info
sub_axes_map(@nospecialize(T::Type{<:SubArray})) = _sub_axes_map(T, IndicesInfo(T))
function _sub_axes_map(@nospecialize(T::Type{<:SubArray}), ::IndicesInfo{N,pdims}) where {N,pdims}
    ntuple(length(pdims)) do i
        if fieldtype(fieldtype(T, :indices), i) <: Base.Slice{OneTo{Int}}
            sz = known_size(parent_type(T), getfield(pdims, i))
            sz === nothing ? StaticInt(i) : StaticSymbol(:parent) => StaticInt(sz)
        else
            StaticInt(i)
        end
    end
end

sub_dimnames_map(@nospecialize T::Type{<:SubArray}) = _sub_dimnames_map(IndicesInfo(T))
function _sub_dimnames_map(::IndicesInfo{N,pdims,cdims}) where {N,pdims,cdims}
    ntuple(length(pdims)) do i
        cdim_i = getfield(cdims, i)
        pdim_i = getfield(pdims, i)
        if cdim_i === 0
            StaticSymbol(:underscore) => StaticInt(0)
        elseif pdim_i isa Int && cdim_i isa Int
            StaticInt(pdim_i)
        else
            StaticSymbol(:underscore) => length(cdim_i)
        end
    end
end

"""
    from_parent_dims(::Type{T}) -> Tuple{Vararg{Union{StaticInt,Tuple{Vararg{StaticInt}}}}}

Returns the mapping from parent dimensions to child dimensions.

!!! Warning
    This method is still experimental and may change without notice.

"""
from_parent_dims(@nospecialize x) = from_parent_dims(typeof(x))
from_parent_dims(@nospecialize T::Type{<:PermutedDimsArray}) = getfield(_permdims(T), 2)
from_parent_dims(@nospecialize T::Type{<:MatAdjTrans}) = (StaticInt(2), StaticInt(1))
@inline function from_parent_dims(@nospecialize T::Type{<:SubArray})
    from_parent_dims(IndicesInfo{ndims(parent_type(T))}(fieldtype(T, :indices)))
end
# TODO do I need to flatten_tuples here?
function from_parent_dims(::IndicesInfo{N,pdims,cdims}) where {N,pdims,cdims}
    ntuple(length(cdims)) do i
        pdim_i = getfield(pdims, i)
        cdim_i = static(getfield(cdims, i))
        pdim_i isa Int ? cdim_i : ntuple(Compat.Returns(cdim_i), length(pdim_i))
    end
end

"""
    has_dimnames(::Type{T}) -> Bool

Returns `true` if `x` has on or more named dimensions. If all dimensions correspond
to `:_`, then `false` is returned.
"""
@inline has_dimnames(x) = static(known_dimnames(x) !== ntuple(Compat.Returns(:_), Val(ndims(x))))

"""
    known_dimnames(::Type{T}) -> Tuple{Vararg{Union{Symbol,Nothing}}}
    known_dimnames(::Type{T}, dim::Union{Int,StaticInt}) -> Union{Symbol,Nothing}

Return the names of the dimensions for `x`. `:_` is used to indicate a dimension does not
have a name.
"""
@inline known_dimnames(x, dim) = _known_dimname(known_dimnames(x), canonicalize(dim))
known_dimnames(x) = known_dimnames(typeof(x))
function known_dimnames(@nospecialize T::Type{<:VecAdjTrans})
    (:_, getfield(known_dimnames(parent_type(T)), 1))
end
function known_dimnames(@nospecialize T::Type{<:Union{MatAdjTrans,PermutedDimsArray}})
    map(GetIndex{false}(known_dimnames(parent_type(T))), to_parent_dims(T))
end

function known_dimnames(@nospecialize T::Type{<:SubArray})
    flatten_tuples(map(Base.Fix1(_known_sub_dimname, known_dimnames(parent_type(T))), sub_dimnames_map(T)))
end
_known_sub_dimname(dn::Tuple, ::StaticInt{dim}) where {dim} = getfield(dn, dim)
function _known_sub_dimname(::Tuple, ::Pair{StaticSymbol{:underscore},StaticInt{N}}) where {N}
    ntuple(Compat.Returns(:_), StaticInt(N))
end

function known_dimnames(::Type{<:ReinterpretArray{T,N,S,A,IsReshaped}}) where {T,N,S,A,IsReshaped}
    pnames = known_dimnames(A)
    if IsReshaped
        if sizeof(S) === sizeof(T)
            return pnames
        elseif sizeof(S) > sizeof(T)
            return (:_, pnames...)
        else
            return tail(pnames)
        end
    else
        return pnames
    end
end

@inline function known_dimnames(@nospecialize T::Type{<:Base.ReshapedArray})
    if ndims(T) === ndims(parent_type(T))
        return known_dimnames(parent_type(T))
    elseif ndims(T) > ndims(parent_type(T))
        return flatten_tuples((known_dimnames(parent_type(T)), ntuple(Compat.Returns(:_), StaticInt(ndims(T) - ndims(parent_type(T))))))
    else
        return ntuple(Compat.Returns(:_), StaticInt(ndims(T)))
    end
end
@inline function known_dimnames(::Type{T}) where {T}
    if is_forwarding_wrapper(T)
        return known_dimnames(parent_type(T))
    else
        return _unknown_dimnames(Base.IteratorSize(T))
    end
end

_unknown_dimnames(::Base.HasShape{N}) where {N} = ntuple(Compat.Returns(:_), StaticInt(N))
_unknown_dimnames(::Any) = (:_,)

@inline function _known_dimname(x::Tuple{Vararg{Any,N}}, dim::CanonicalInt) where {N}
    # we cannot have `@boundscheck`, else this will depend on bounds checking being enabled
    (dim > N || dim < 1) && return :_
    return @inbounds(x[dim])
end
@inline _inbounds_known_dimname(x, dim) = @inbounds(_known_dimname(x, dim))

"""
    dimnames(x) -> Tuple{Vararg{Union{Symbol,StaticSymbol}}}
    dimnames(x, dim::Union{Int,StaticInt}) -> Union{Symbol,StaticSymbol}

Return the names of the dimensions for `x`. `:_` is used to indicate a dimension does not
have a name.
"""
@inline dimnames(x, dim) = _dimname(dimnames(x), canonicalize(dim))
@inline function dimnames(x::Union{PermutedDimsArray,MatAdjTrans})
    map(GetIndex{false}(dimnames(parent(x))), to_parent_dims(x))
end

function dimnames(x::SubArray)
    flatten_tuples(map(Base.Fix1(_sub_dimname, dimnames(parent(x))), sub_dimnames_map(typeof(x))))
end
_sub_dimname(dn::Tuple, ::StaticInt{dim}) where {dim} = getfield(dn, dim)
function _sub_dimname(::Tuple, ::Pair{StaticSymbol{:underscore},StaticInt{N}}) where {N}
    ntuple(Compat.Returns(static(:_)), StaticInt(N))
end

dimnames(x::VecAdjTrans) = (static(:_), getfield(dimnames(parent(x)), 1))
@inline function dimnames(x::ReinterpretArray{T,N,S,A,IsReshaped}) where {T,N,S,A,IsReshaped}
    pnames = dimnames(parent(x))
    if IsReshaped
        if sizeof(S) === sizeof(T)
            return pnames
        elseif sizeof(S) > sizeof(T)
            return flatten_tuples((static(:_), pnames))
        else
            return tail(pnames)
        end
    else
        return pnames
    end
end
@inline function dimnames(x::Base.ReshapedArray)
    p = parent(x)
    if ndims(x) === ndims(p)
        return dimnames(p)
    elseif ndims(x) > ndims(p)
        return flatten_tuples((dimnames(p), ntuple(Compat.Returns(static(:_)), StaticInt(ndims(x) - ndims(p)))))
    else
        return ntuple(Compat.Returns(static(:_)), StaticInt(ndims(x)))
    end
end
@inline function dimnames(x::X) where {X}
    if is_forwarding_wrapper(X)
        return dimnames(parent(x))
    else
        return ntuple(Compat.Returns(static(:_)), StaticInt(ndims(x)))
    end
end
@inline function _dimname(x::Tuple{Vararg{Any,N}}, dim::CanonicalInt) where {N}
    # we cannot have `@boundscheck`, else this will depend on bounds checking being enabled
    # for calls such as `dimnames(view(x, :, 1, :))`
    (dim > N || dim < 1) && return static(:_)
    return @inbounds(x[dim])
end
@inline _inbounds_dimname(x, dim) = @inbounds(_dimname(x, dim))

"""
    to_dims(x, dim) -> Union{Int,StaticInt}

This returns the dimension(s) of `x` corresponding to `dim`.
"""
to_dims(x, dim::Colon) = dim
to_dims(x, @nospecialize(dim::CanonicalInt)) = dim
to_dims(x, dim::Integer) = Int(dim)
to_dims(x, dim::Union{StaticSymbol,Symbol}) = _to_dim(dimnames(x), dim)
function to_dims(x, dims::Tuple{Vararg{Any,N}}) where {N}
    eachop(_to_dims, ntuple(static, StaticInt(N)), dimnames(x), dims)
end
@inline _to_dims(x::Tuple, d::Tuple, n::StaticInt{N}) where {N} = _to_dim(x, getfield(d, N))
@inline function _to_dim(x::Tuple, d::Union{Symbol,StaticSymbol})
    i = find_first_eq(d, x)
    i === nothing && throw(DimensionMismatch("dimension name $(d) not found"))
    return i
end

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
@generated function find_all_dimnames(x::Tuple{Vararg{Any,ND}}, nd::Tuple{Vararg{Any,NI}}, inds::Tuple, default) where {ND,NI}
    if NI === 0
        return :(())
    else
        out = Expr(:block, Expr(:(=), :names_found, 0))
        t = Expr(:tuple)
        for i in 1:ND
            index_i = Symbol(:index_, i)
            val_i = Symbol(:val_, i)
            push!(t.args, val_i)
            push!(out.args, quote
                $index_i = find_first_eq(getfield(x, $i), nd)
                if $index_i === nothing
                    $val_i = default
                else
                    $val_i = @inbounds(inds[$index_i])
                    names_found += 1
                end
            end)
        end
        return quote
            $out
            @boundscheck names_found === $NI || error("Not all keywords matched dimension names.")
            return $t
        end
    end
end
