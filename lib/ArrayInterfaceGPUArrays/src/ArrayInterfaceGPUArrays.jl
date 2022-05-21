module ArrayInterfaceGPUArrays

using Adapt
using ArrayInterfaceCore
using GPUArrays

ArrayInterfaceCore.fast_scalar_indexing(::Type{<:GPUArrays.AbstractGPUArray}) = false
@inline ArrayInterfaceCore.allowed_getindex(x::GPUArrays.AbstractGPUArray, i...) = GPUArrays.@allowscalar(x[i...])
@inline ArrayInterfaceCore.allowed_setindex!(x::GPUArrays.AbstractGPUArray, v, i...) = (GPUArrays.@allowscalar(x[i...] = v))

function Base.setindex(x::GPUArrays.AbstractGPUArray, v, i::Int)
    _x = copy(x)
    ArrayInterfaceCore.allowed_setindex!(_x, v, i)
    return _x
end

function ArrayInterfaceCore.restructure(x::GPUArrays.AbstractGPUArray, y)
    reshape(Adapt.adapt(ArrayInterfaceCore.parameterless_type(x), y), Base.size(x)...)
end

function ArrayInterfaceCore.lu_instance(A::GPUArrays.AbstractGPUMatrix{T}) where {T}
    qr(similar(A, 1, 1))
end

# Doesn't do much, but makes a gigantic change to the dependency chain.
# ArrayInterface.device(::Type{<:GPUArrays.AbstractGPUArray}) = ArrayInterface.GPU()

end