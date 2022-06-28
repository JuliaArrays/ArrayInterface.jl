

# linear index into linear collection
@inline function indices_to_dimensions(::IndicesInfo{(1,),NS,nothing}, ::StaticInt{1}) where {NS}
    (_add_dims(1, getfield(NS, 1)),), (StaticInt(1),)
end
@inline function indices_to_dimensions(::IndicesInfo{(1,),NS,IS}, ::StaticInt{1}) where {NS,IS}
    (_add_dims(1, getfield(NS, 1)),), (StaticInt(1),)
end
@inline function indices_to_dimensions(::IndicesInfo{(1,),NS,nothing}, n::StaticInt{N}) where {NS,N}
    (_add_dims(1, getfield(NS, 1)),), (:,)
end
@inline function indices_to_dimensions(::IndicesInfo{(1,),NS,IS}, n::StaticInt{N}) where {NS,N,IS}
    (_add_dims(1, getfield(NS, 1)),), ntuple(static, n)
end
@inline function indices_to_dimensions(::IndicesInfo{NI,NS,nothing}, n::StaticInt{N}) where {NI,NS,N}
    _accum_dims(NS), sum(NI) > N ? _replace_trailing(n, _accum_dims(NI)) : _accum_dims(NI)
end
@inline function indices_to_dimensions(::IndicesInfo{NI,NS,IS}, n::StaticInt{N}) where {NI,NS,IS,N}
    ndims_indices = sum(NI)
    if ndims_indices === N
        return _accum_dims(NS), _accum_dims(NI)
    else
        splat_map = ntuple(Base.Fix2(_replace_splat, max(0, N - ndims_indices + 1)) ∘ ==(IS), length(NI))
        return _accum_dims(map(*, NS, splat_map)), _accum_dims(map(*, NI, splat_map))
    end
end

_replace_splat(is_splat::Bool, n::Int) = is_splat ? n : 1
_replace_trailing(::StaticInt{N}, dim::StaticInt{D}) where {N,D} = N < D ? StaticInt(0) : dim
_replace_trailing(n::StaticInt{N}, dims::Tuple) where {N} = map(Base.Fix1(_replace_trailing, n), dims)
_accum_dims(dims::Tuple) = map(_add_dims, cumsum(dims), dims)
@inline function _add_dims(dim::Int, n::Int)
    if n === 0
        return StaticInt(0)
    elseif n === 1
        return StaticInt(dim)
    else
        return ntuple(static ∘ Base.Fix1(+, dim - n), n)
    end
end

dimsmap(x) = dimsmap(typeof(x))
function dimsmap(@nospecialize T::Type{<:SubArray})
    dimsin, dimsout = indices_to_dimensions(IndicesInfo(fieldtype(T, :indices)), StaticInt(ndims(parent_type(T))))
    map(MappedIndex, dimsin, dimsout, ntuple(static, length(dimsin)))
end
@inline function to_parent_dims(@nospecialize T::Type{<:SubArray})
    dimsin, dimsout = indices_to_dimensions(IndicesInfo(fieldtype(T, :indices)), StaticInt(ndims(parent_type(T))))
    flatten_tuples(map(_to_parent_dim, dimsin, dimsout))
end
_to_parent_dim(::StaticInt{0}, pdim) = ()
_to_parent_dim(::StaticInt{cdim}, pdim) where {cdim} = (pdim,)
_to_parent_dim(cdims::Tuple, pdim) = ntuple(Compat.Returns(pdim), length(cdims))

@inline function from_parent_dims(@nospecialize T::Type{<:SubArray})
    dimsin, dimsout = indices_to_dimensions(IndicesInfo(fieldtype(T, :indices)), StaticInt(ndims(parent_type(T))))
    flatten_tuples(map(_from_parent_dim, dimsin, dimsout))
end
_from_parent_dim(cdim, ::StaticInt) = (cdim,)
_from_parent_dim(cdim, pdims::Tuple) = ntuple(Compat.Returns(cdim), length(pdims))

function throw_dim_error(@nospecialize(x), @nospecialize(dim))
    throw(DimensionMismatch("$x does not have dimension corresponding to $dim"))
end

function _permdims(::Type{<:PermutedDimsArray{<:Any,<:Any,I1,I2}}) where {I1,I2}
    (map(static, I1), map(static, I2))
end
#### TODO maybe document
dimperm(@nospecialize x) = dimperm(typeof(x))
dimperm(@nospecialize T::Type{<:MatAdjTrans}) = (StaticInt(2), StaticInt(1))
dimperm(@nospecialize T::Type{<:PermutedDimsArray}) = getfield(_permdims(T), 1)

invdimperm(@nospecialize x) = invdimperm(typeof(x))
invdimperm(@nospecialize T::Type{<:PermutedDimsArray}) = getfield(_permdims(T), 2)
invdimperm(@nospecialize T::Type{<:MatAdjTrans}) = (StaticInt(2), StaticInt(1))

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
    map(GetIndex{false}(known_dimnames(parent_type(T))), dimperm(T))
end

known_dimnames(@nospecialize T::Type{<:SubArray}) = flatten_tuples(map(Base.Fix1(_sub_known_dimnames, known_dimnames(parent_type(T))), dimsmap(T)))
_sub_known_dimnames(@nospecialize(dnames::Tuple), @nospecialize(mi::MappedIndex{StaticInt{0}})) = ()
_sub_known_dimnames(@nospecialize(dnames::Tuple), @nospecialize(mi::MappedIndex{<:Tuple})) = ntuple(Compat.Returns(:_), Val{nfields(dimsin(mi))}())
_sub_known_dimnames(@nospecialize(dnames::Tuple), @nospecialize(mi::MappedIndex{<:StaticInt})) = __sub_known_dimnames(dnames, dimsout(mi))
__sub_known_dimnames(@nospecialize(dnames::Tuple), @nospecialize(dimsout::Tuple)) = :_
__sub_known_dimnames(dnames::Tuple, ::StaticInt{dimout}) where {dimout} = getfield(dnames, dimout)

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
    map(GetIndex{false}(dimnames(parent(x))), dimperm(x))
end

dimnames(x::SubArray) = flatten_tuples(map(Base.Fix1(_sub_dimnames, dimnames(parent(x))), dimsmap(x)))
_sub_dimnames(@nospecialize(dnames::Tuple), @nospecialize(mi::MappedIndex{StaticInt{0}})) = ()
_sub_dimnames(@nospecialize(dnames::Tuple), @nospecialize(mi::MappedIndex{<:Tuple})) = ntuple(Compat.Returns(static(:_)), Val{nfields(dimsin(mi))}())
_sub_dimnames(@nospecialize(dnames::Tuple), @nospecialize(mi::MappedIndex{<:StaticInt})) = __sub_dimnames(dnames, dimsout(mi))
__sub_dimnames(@nospecialize(dnames::Tuple), @nospecialize(dimsout::Tuple)) = static(:_)
__sub_dimnames(dnames::Tuple, ::StaticInt{dimout}) where {dimout} = getfield(dnames, dimout)

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
