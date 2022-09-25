module ArrayInterfaceGPUArrays

using Adapt
using ArrayInterfaceCore
using GPUArraysCore
using LinearAlgebra: lu

ArrayInterfaceCore.fast_scalar_indexing(::Type{<:GPUArraysCore.AbstractGPUArray}) = false
@inline ArrayInterfaceCore.allowed_getindex(x::GPUArraysCore.AbstractGPUArray, i...) = GPUArraysCore.@allowscalar(x[i...])
@inline ArrayInterfaceCore.allowed_setindex!(x::GPUArraysCore.AbstractGPUArray, v, i...) = (GPUArraysCore.@allowscalar(x[i...] = v))

function Base.setindex(x::GPUArraysCore.AbstractGPUArray, v, i::Int)
    _x = copy(x)
    ArrayInterfaceCore.allowed_setindex!(_x, v, i)
    return _x
end

function ArrayInterfaceCore.restructure(x::GPUArraysCore.AbstractGPUArray, y)
    reshape(Adapt.adapt(ArrayInterfaceCore.parameterless_type(x), y), Base.size(x)...)
end

function ArrayInterfaceCore.lu_instance(A::GPUArraysCore.AbstractGPUMatrix{T}) where {T}
    lu(Adapt.adapt(ArrayInterfaceCore.parameterless_type(A), ones(T, 0, 0)))
end

# Doesn't do much, but makes a gigantic change to the dependency chain.
# ArrayInterface.device(::Type{<:GPUArraysCore.AbstractGPUArray}) = ArrayInterface.GPU()

end
