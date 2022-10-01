

_init_dimsmap(x) = _init_dimsmap(IndicesInfo(x))
function _init_dimsmap(@nospecialize info::IndicesInfo)
    pdims = parentdims(info)
    cdims = childdims(info)
    ntuple(i -> static(getfield(pdims, i)), length(pdims)),
    ntuple(i -> static(getfield(cdims, i)), length(pdims))
end

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
to_parent_dims(info::IndicesInfo) = flatten_tuples(map(_to_pdim, map_indices_info(info)))
_to_pdim(::Tuple{StaticInt,Any,StaticInt{0}}) = ()
_to_pdim(x::Tuple{StaticInt,Any,StaticInt{cdim}}) where {cdim} = getfield(x, 2)
_to_pdim(x::Tuple{StaticInt,Any,Tuple}) = (ntuple(Compat.Returns(getfield(x, 2)), length(getfield(x, 3))),)
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
function sub_axes_map(@nospecialize(T::Type{<:SubArray}))
    map(Base.Fix1(_sub_axis_map, T), map_indices_info(IndicesInfo(T)))
end
function _sub_axis_map(@nospecialize(T::Type{<:SubArray}), x::Tuple{StaticInt{index},Any,Any}) where {index}
    if fieldtype(fieldtype(T, :indices), index) <: Base.Slice{OneTo{Int}}
        sz = known_size(parent_type(T), getfield(x, 2))
        return sz === nothing ? StaticInt(index) : StaticInt(1):StaticInt(sz)
    else
        return StaticInt(index)
    end
end

function map_indices_info(@nospecialize info::IndicesInfo)
    pdims = parentdims(info)
    cdims = childdims(info)
    ntuple(i -> (static(i), static(getfield(pdims, i)), static(getfield(cdims, i))), length(pdims))
end
function sub_dimnames_map(dnames::Tuple, imap::Tuple)
    flatten_tuples(map(Base.Fix1(_to_dimname, dnames), imap))
end
@inline function _to_dimname(dnames::Tuple, x::Tuple{StaticInt,PD,CD}) where {PD,CD}
    if CD <: StaticInt{0}
        return ()
    elseif CD <: Tuple
        return ntuple(Compat.Returns(static(:_)), StaticInt(known_length(CD)))
    elseif PD <: StaticInt{0} || PD <: Tuple
        return static(:_)
    else
        return getfield(dnames, known(PD))
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
function from_parent_dims(@nospecialize(info::IndicesInfo))
    pdims = parentdims(info)
    cdims = childdims(info)
    ntuple(length(cdims)) do i
        pdim_i = getfield(pdims, i)
        cdim_i = static(getfield(cdims, i))
        pdim_i isa Int ? cdim_i : ntuple(Compat.Returns(cdim_i), length(pdim_i))
    end
end
