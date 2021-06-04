fast_scalar_indexing(::Type{<:CuArrays.CuArray}) = false
@inline allowed_getindex(x::CuArrays.CuArray, i...) = CuArrays.@allowscalar(x[i...])
@inline function allowed_setindex!(x::CuArrays.CuArray, v, i...)
    return (CuArrays.@allowscalar(x[i...] = v))
end

function Base.setindex(x::CuArrays.CuArray, v, i::Int)
    _x = copy(x)
    allowed_setindex!(_x, v, i)
    return _x
end

function restructure(x::CuArrays.CuArray, y)
    return reshape(Adapt.adapt(parameterless_type(x), y), Base.size(x)...)
end

device(::Type{<:CuArrays.CuArray}) = GPU()

