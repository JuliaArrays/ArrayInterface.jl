@inline function StrideIndex{N,R,C}(a::A) where {N,R,C,A}
    return StrideIndex{N,R,C}(static_strides(a), offsets(a))
end
@inline function StrideIndex(a::A) where {A}
    return StrideIndex{ndims(A),known(stride_rank(A)),known(contiguous_axis(A))}(a)
end

## getindex
@propagate_inbounds Base.getindex(x::ArrayIndex, i::IntType, ii::IntType...) = x[NDIndex(i, ii...)]

@inline function Base.getindex(x::StrideIndex{N}, i::AbstractCartesianIndex) where {N}
    return _strides2int(offsets(x), static_strides(x), Tuple(i)) + static(1)
end

@generated function _strides2int(o::O, s::S, i::I) where {O,S,I}
    N = known_length(S)
    out = :()
    for i in 1:N
        tmp = :(((getfield(i, $i) - getfield(o, $i)) * getfield(s, $i)))
        out = ifelse(i === 1, tmp, :($out + $tmp))
    end
    return Expr(:block, Expr(:meta, :inline), out)
end