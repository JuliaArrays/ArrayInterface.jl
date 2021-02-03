


function reinterpret_size1(::StaticInt{p}, ::Type{S}, ::Type{T}) where {p,S,T}
    return StaticInt(_reinterpret_size1(p, sizeof(S), sizeof(T)))
end
function reinterpret_size1(p::Int, ::Type{S}, ::Type{T}) where {S,T}
    return _reinterpret_size1(p, sizeof(S), sizeof(T))
end
@pure function _reinterpret_size1(p::Int, s::Int, t::Int)
    return Core.Intrinsics.checked_sdiv_int(Core.Intrinsics.mul_int(p, s), t)
end


function known_size(::Type{ReinterpretArray{T,N,S,A,false}}) where {T,N,S,A}
    psize = known_size(A)
    if first(psize) === nothing
        return psize
    else
        return (reinterpret_size1(first(psize), S, T), tail(psize)...)
    end
end
function known_size(::Type{ReinterpretArray{T,N,S,A,true}}) where {T,N,S,A}
    psize = known_size(A)
    if sizeof(S) < sizeof(T)
        return tail(psize)
    elseif first(psize) === nothing
        return psize
    elseif sizeof(S) > sizeof(T)
        return (reinterpret_size1(first(psize), S, T), tail(psize)...)
    else
        return psize
    end
end
known_size(::Type{ReinterpretArray{T,0,S,A,false}}) where {T,S,A} = ()

function size(a::ReinterpretArray{T,N,S,A,false}) where {T,N,S,A}
    psize = size(parent(a))
    return (reinterpret_size1(first(psize), S, T), tail(psize)...)
end
function size(a::ReinterpretArray{T,N,S,A,true}) where {T,N,S,A}
    psize = size(parent(a))
    if sizeof(S) > sizeof(T)
        return (reinterpret_size1(first(psize), S, T), tail(psize)...)
    elseif sizeof(S) < sizeof(T)
        return tail(psize)
    else
        return psize
    end
end
size(::ReinterpretArray{T,0,S,A,false}) where {T,S,A} = ()

axes(a::Base.ReinterpretArray, d) = Base.axes(a, Int(to_dims(a, d)))
axes(a::Base.ReinterpretArray) = Base.axes(a)


function strides(a::ReinterpretArray)
    defines_strides(parent_type(a)) || ArgumentError("Parent must be strided.") |> throw
    return size_to_strides(One(), size(a)...)
end
@inline size_to_strides(s, d, sz...) = (s, size_to_strides(s * d, sz...)...)
size_to_strides(s, d) = (s,)
size_to_strides(s) = ()

