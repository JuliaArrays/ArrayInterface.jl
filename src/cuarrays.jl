fast_scalar_indexing(::Type{<:CuArrays.CuArray}) = false
@inline allowed_getindex(x::CuArrays.CuArray,i...) = CuArrays.@allowscalar(x[i...])
@inline allowed_setindex!(x::CuArrays.CuArray,v,i...) = (CuArrays.@allowscalar(x[i...] = v))

function Base.setindex(x::CuArrays.CuArray,v,i::Int)
  _x = copy(x)
  allowed_setindex!(_x,v,i)
  _x
end

function restructure(x::CuArrays.CuArray,y)
  reshape(Adapt.adapt(parameterless_type(x),y), Base.size(x)...)
end

Device(::Type{<:CuArrays.CuArray}) = GPU()

