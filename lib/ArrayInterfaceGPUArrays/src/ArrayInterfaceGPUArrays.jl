module ArrayInterfaceGPUArrays

using Adapt
using ArrayInterface
using GPUArrays

ArrayInterface.fast_scalar_indexing(::Type{<:GPUArrays.AbstractGPUArray}) = false
@inline ArrayInterface.allowed_getindex(x::GPUArrays.AbstractGPUArray, i...) = CUDA.@allowscalar(x[i...])
@inline ArrayInterface.allowed_setindex!(x::GPUArrays.AbstractGPUArray, v, i...) = (CUDA.@allowscalar(x[i...] = v))

function Base.setindex(x::GPUArrays.AbstractGPUArray, v, i::Int)
    _x = copy(x)
    ArrayInterface.allowed_setindex!(_x, v, i)
    return _x
end

function ArrayInterface.restructure(x::GPUArrays.AbstractGPUArray, y)
    reshape(Adapt.adapt(ArrayInterface.parameterless_type(x), y), Base.size(x)...)
end

function ArrayInterface.lu_instance(A::GPUArrays.AbstractGPUMatrix{T}) where {T}
    qr(similar(A, 1, 1))
end

# Doesn't do much, but makes a gigantic change to the dependency chain.
# ArrayInterface.device(::Type{<:GPUArrays.AbstractGPUArray}) = ArrayInterface.GPU()

end