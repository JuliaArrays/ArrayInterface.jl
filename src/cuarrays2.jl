fast_scalar_indexing(::Type{<:CUDA.CuArray}) = false
@inline allowed_getindex(x::CUDA.CuArray,i...) = CUDA.@allowscalar(x[i...])
@inline allowed_setindex!(x::CUDA.CuArray,v,i...) = (CUDA.@allowscalar(x[i...] = v))

function Base.setindex(x::CUDA.CuArray,v,i::Int)
  _x = copy(x)
  allowed_setindex!(_x,v,i)
  _x
end

function restructure(x::CUDA.CuArray,y)
  reshape(Adapt.adapt(parameterless_type(x),y), Base.size(x)...)
end

Device(::Type{<:CUDA.CuArray}) = GPU()

