
"""
    axes_types(::Type{T}, dim)

Returns the axis type along dimension `dim`.
"""
axes_types(x, dim) = axes_types(typeof(x), dim)
@inline axes_types(::Type{T}, dim) where {T} = axes_types(T, to_dims(T, dim))
@inline function axes_types(::Type{T}, dim::StaticInt{D}) where {T,D}
    if D > ndims(T)
        return OptionallyStaticUnitRange{One,One}
    else
        return _get_tuple(axes_types(T), dim)
    end
end
@inline function axes_types(::Type{T}, dim::Int) where {T}
    if dim > ndims(T)
        return OptionallyStaticUnitRange{One,One}
    else
        return axes_types(T).parameters[dim]
    end
end

"""
    axes_types(::Type{T}) -> Type

Returns the type of the axes for `T`
"""
axes_types(x) = axes_types(typeof(x))
axes_types(::Type{T}) where {T<:Array} = Tuple{Vararg{OneTo{Int},ndims(T)}}
function axes_types(::Type{T}) where {T}
    if parent_type(T) <: T
        return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},ndims(T)}}
    else
        return axes_types(parent_type(T))
    end
end
axes_types(::Type{LinearIndices{N,R}}) where {N,R} = R
axes_types(::Type{CartesianIndices{N,R}}) where {N,R} = R
function axes_types(::Type{T}) where {T<:VecAdjTrans}
    return Tuple{OptionallyStaticUnitRange{One,One},axes_types(parent_type(T), One())}
end
function axes_types(::Type{T}) where {T<:MatAdjTrans}
    return eachop_tuple(_get_tuple, axes_types(parent_type(T)); iterator=to_parent_dims(T))
end
function axes_types(::Type{T}) where {T<:PermutedDimsArray}
    return eachop_tuple(_get_tuple, axes_types(parent_type(T)); iterator=to_parent_dims(T))
end
function axes_types(::Type{T}) where {T<:AbstractRange}
    if known_length(T) === nothing
        return Tuple{OptionallyStaticUnitRange{One,Int}}
    else
        return Tuple{OptionallyStaticUnitRange{One,StaticInt{known_length(T)}}}
    end
end
function axes_types(::Type{T}) where {N,T<:Base.ReshapedArray{<:Any,N}}
    return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},N}}
end

_int_or_static_int(::Nothing) = Int
_int_or_static_int(x::Int) = StaticInt{x}

@inline function axes_types(::Type{T}) where {N,P,I,T<:SubArray{<:Any,N,P,I}}
    return eachop_tuple(_sub_axis_type, T; iterator=to_parent_dims(T))
end
@inline function _sub_axis_type(::Type{A}, dim::StaticInt) where {T,N,P,I,A<:SubArray{T,N,P,I}}
    return OptionallyStaticUnitRange{
        _int_or_static_int(known_first(axes_types(P, dim))),
        _int_or_static_int(known_length(_get_tuple(I, dim)))
    }
end

function axes_types(::Type{R}) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    if _is_reshaped(R)
        if sizeof(S) === sizeof(T)
            return axes_types(A)
        elseif sizeof(S) > sizeof(T)
            return eachop_tuple(_reshaped_axis_type, R; iterator=to_parent_dims(R))
        else
            return eachop_tuple(axes_types, A; iterator=to_parent_dims(R))
        end
    else
        return eachop_tuple(_non_reshaped_axis_type, R; iterator=to_parent_dims(R))
    end
end

function _reshaped_axis_type(::Type{R}, dim::StaticInt) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    return axes_types(parent_type(R), dim)
end
function _reshaped_axis_type(::Type{R}, dim::Zero) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    return OptionallyStaticUnitRange{One,StaticInt{div(sizeof(S), sizeof(T))}}
end

function _non_reshaped_axis_type(::Type{R}, dim::StaticInt) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    return axes_types(parent_type(R), dim)
end
function _non_reshaped_axis_type(::Type{R}, dim::One) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    paxis = axes_types(A, dim)
    len = known_length(paxis)
    if len === nothing
        raxis = OptionallyStaticUnitRange{One,Int}
    else
        raxis = OptionallyStaticUnitRange{One,StaticInt{div(len * sizeof(S), sizeof(T))}}
    end
    return similar_type(paxis, Int, raxis)
end

#=
    similar_type(orignal_type, new_data_type)
=#
similar_type(::Type{OneTo{Int}}, ::Type{Int}, ::Type{OneTo{Int}}) = OneTo{Int}
similar_type(::Type{OneTo{Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,Int}}) = OneTo{Int}
similar_type(::Type{OneTo{Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}) where {N} = OptionallyStaticUnitRange{One,StaticInt{N}}

similar_type(::Type{OptionallyStaticUnitRange{One,Int}}, ::Type{Int}, ::Type{OneTo{Int}}) = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,Int}}) = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,Int}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}) where {N} = OptionallyStaticUnitRange{One,StaticInt{N}}

similar_type(::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}, ::Type{Int}, ::Type{OneTo{Int}}) where {N} = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,StaticInt{N}}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,Int}}) where {N} = OptionallyStaticUnitRange{One,Int}
similar_type(::Type{OptionallyStaticUnitRange{One,StaticInt{N1}}}, ::Type{Int}, ::Type{OptionallyStaticUnitRange{One,StaticInt{N2}}}) where {N1,N2} = OptionallyStaticUnitRange{One,StaticInt{N2}}

"""
    axes(A, d)

Return a valid range that maps to each index along dimension `d` of `A`.
"""
axes(a, dim) = axes(a, to_dims(a, dim))
function axes(a::A, dim::Integer) where {A}
    if parent_type(A) <: A
        return Base.axes(a, Int(dim))
    else
        return axes(parent(a), to_parent_dims(A, dim))
    end
end
axes(A::SubArray, dim::Integer) = Base.axes(A, Int(dim))  # TODO implement ArrayInterface version
axes(A::ReinterpretArray, dim::Integer) = Base.axes(A, Int(dim))  # TODO implement ArrayInterface version
axes(A::Base.ReshapedArray, dim::Integer) = Base.axes(A, Int(dim))  # TODO implement ArrayInterface version

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
axes(A::PermutedDimsArray) = permute(axes(parent(A)), to_parent_dims(A))
function axes(A::Union{Transpose,Adjoint})
    p = parent(A)
    return (axes(p, StaticInt(2)), axes(p, One()))
end
axes(A::SubArray) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::ReinterpretArray) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::Base.ReshapedArray) = Base.axes(A)  # TODO implement ArrayInterface version

