
"""
    axes_types(::Type{T}, dim)

Returns the axis type along dimension `dim`.
"""
axes_types(x, dim) = axes_types(typeof(x), dim)
@inline axes_types(::Type{T}, dim) where {T} = axes_types(T, to_dims(T, dim))
@inline function axes_types(::Type{T}, dim::StaticInt) where {T}
    if dim > ndims(T)
        return OptionallyStaticUnitRange{One,One}
    else
        return _get_tuple(axes_types(T), dim)
    end
end
@inline function axes_types(::Type{T}, dim::Integer) where {T}
    if dim > ndims(T)
        return OptionallyStaticUnitRange{One,One}
    else
        return axes_types(T).parameters[Int(dim)]
    end
end

"""
    axes_types(::Type{T}) -> Type

Returns the type of the axes for `T`
"""
axes_types(x) = axes_types(typeof(x))
function axes_types(::Type{T}) where {T}
    if parent_type(T) <: T
        return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},ndims(T)}}
    else
        return eachop_tuple(axes_types, parent_type(T), to_parent_dims(T))
    end
end
function axes_types(::Type{T}) where {T<:MatAdjTrans}
    return eachop_tuple(_get_tuple, axes_types(parent_type(T)), to_parent_dims(T))
end
function axes_types(::Type{T}) where {T<:PermutedDimsArray}
    return eachop_tuple(_get_tuple, axes_types(parent_type(T)), to_parent_dims(T))
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
axes(A::SubArray, dim::Integer) = Base.axes(A, dim)  # TODO implement ArrayInterface version
axes(A::ReinterpretArray, dim::Integer) = Base.axes(A, dim)  # TODO implement ArrayInterface version
axes(A::Base.ReshapedArray, dim::Integer) = Base.axes(A, dim)  # TODO implement ArrayInterface version

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
axes(A::Union{Transpose,Adjoint}) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::SubArray) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::ReinterpretArray) = Base.axes(A)  # TODO implement ArrayInterface version
axes(A::Base.ReshapedArray) = Base.axes(A)  # TODO implement ArrayInterface version

