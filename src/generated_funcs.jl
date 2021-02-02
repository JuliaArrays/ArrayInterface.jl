
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

@generated function known_offsets(::Type{T}) where {T}
    out = Expr(:tuple)
    for p in axes_types(T).parameters
        push!(out.args, known_first(p))
    end
    return out
end


