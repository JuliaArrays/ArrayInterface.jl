
if VERSION ≥ v"1.6.0-DEV.1581"
    _is_reshaped(::Type{ReinterpretArray{T,N,S,A,true}}) where {T,N,S,A} = True()
    _is_reshaped(::Type{ReinterpretArray{T,N,S,A,false}}) where {T,N,S,A} = False()
else
    _is_reshaped(::Type{ReinterpretArray{T,N,S,A}}) where {T,N,S,A} = False()
end

# this handles resizing the first dimension for ReinterpretArray
resize_reinterpreted(::Nothing, ::Type{S}, ::Type{T}) where {S,T} = nothing
function resize_reinterpreted(::StaticInt{p}, ::Type{S}, ::Type{T}) where {p,S,T}
    return StaticInt(_resize_reinterpreted(p, S, T))
end
function resize_reinterpreted(p::Int, ::Type{S}, ::Type{T}) where {S,T}
    return _resize_reinterpreted(p, S, T)
end
@pure function _resize_reinterpreted(p::Int, ::Type{S}, ::Type{T})::Int where {S,T}
    # div(p * sizeof(S), sizeof(T))
    return Core.Intrinsics.checked_sdiv_int(
        Core.Intrinsics.mul_int(p, Core.sizeof(S)), Core.sizeof(T)
    )
end


###
### size/known_size
###
size(::ReinterpretArray{T,0}) where {T} = ()
size(x::ReinterpretArray) = _reinterpret_size(x, _is_reshaped(typeof(x)))
function _reinterpret_size(x::ReinterpretArray, ::False)
    return eachop(_size_dim, x, nstatic(Val(ndims(x))))
end
function _reinterpret_size(x::ReinterpretArray, ::True)
    if sizeof(S) === sizeof(T)
        return size(parent(x))
    elseif sizeof(S) > sizeof(T)
        return eachop(_size_dim, x, nstatic(Val(ndims(x))))
    else
        return tail(size(parent(x)))
    end
end
function _size_dim(x::ReinterpretArray{T,N,S}, dim::One) where {T,N,S}
    return resize_reinterpreted(size(parent(x), dim), S, T)
end
_size_dim(x::ReinterpretArray{T,N,S}, dim::StaticInt) where {T,N,S} = size(parent(x), dim)


function known_size(::Type{A}) where {A<:ReinterpretArray}
    return _reinterpret_known_size(A, _is_reshaped(A))
end
function _reinterpret_known_size(::Type{A}, ::False) where {T,N,S,A<:ReinterpretArray{T,N,S}}
    return eachop(_known_size_dim, A, nstatic(Val(N)))
end
function _reinterpret_known_size(::Type{A}, ::True) where {T,N,S,A<:ReinterpretArray{T,N,S}}
    if sizeof(S) === sizeof(T)
        return known_size(parent_type(A))
    elseif sizeof(S) > sizeof(T)
        return eachop(_known_size_dim, A, nstatic(Val(N)))
    else
        return tail(known_size(parent_type(A)))
    end
end
function _known_size_dim(::Type{<:ReinterpretArray{T,N,S,A}}, dim::One) where {T,N,S,A}
    return resize_reinterpreted(known_size(A, dim), S, T)
end
function _known_size_dim(::Type{<:ReinterpretArray{T,N,S,A}}, dim::StaticInt) where {T,N,S,A}
    return known_size(A, dim)
end



axes(a::Base.ReinterpretArray, d) = Base.axes(a, Int(to_dims(a, d)))
axes(a::Base.ReinterpretArray) = Base.axes(a)


function strides(a::ReinterpretArray)
    defines_strides(parent_type(a)) || ArgumentError("Parent must be strided.") |> throw
    return size_to_strides(One(), size(a)...)
end
@inline size_to_strides(s, d, sz...) = (s, size_to_strides(s * d, sz...)...)
size_to_strides(s, d) = (s,)
size_to_strides(s) = ()

# contiguous_if_one(::StaticInt{1}) = StaticInt{1}()
# contiguous_if_one(::Any) = StaticInt{-1}()
function contiguous_axis(::Type{R}) where {T,N,S,A<:Array{S},R<:ReinterpretArray{T,N,S,A}}
    if isbitstype(S)
        return One()
    else
        return nothing
    end
end

function stride_rank(::Type{R}) where {T,N,S,A,R<:Base.ReinterpretArray{T,N,S,A}}
    defines_strides(A) || ArgumentError("Parent must be strided.") |> throw
    return nstatic(Val(N))
end
contiguous_batch_size(::Type{<:Base.ReinterpretArray{T,N,S,A}}) where {T,N,S,A} = Zero()

@inline function known_length(::Type{T}) where {T <: Base.ReinterpretArray}
    return resize_reinterpreted(known_length(parent_type(T)), eltype(parent_type(T)), eltype(T))
end

