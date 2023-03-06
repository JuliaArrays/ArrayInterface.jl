module ArrayInterfaceGPUArraysCoreExt


if isdefined(Base, :get_extension)
    using Adapt
    using ArrayInterface
    using LinearAlgebra: lu
    import GPUArraysCore
else
    using Adapt # Will cause problems for relocatability.
    using ..ArrayInterface
    using ..LinearAlgebra: lu
    import ..GPUArraysCore
end

ArrayInterface.fast_scalar_indexing(::Type{<:GPUArraysCore.AbstractGPUArray}) = false
@inline ArrayInterface.allowed_getindex(x::GPUArraysCore.AbstractGPUArray, i...) = GPUArraysCore.@allowscalar(x[i...])
@inline ArrayInterface.allowed_setindex!(x::GPUArraysCore.AbstractGPUArray, v, i...) = (GPUArraysCore.@allowscalar(x[i...] = v))

function Base.setindex(x::GPUArraysCore.AbstractGPUArray, v, i::Int)
    _x = copy(x)
    ArrayInterface.allowed_setindex!(_x, v, i)
    return _x
end

function ArrayInterface.restructure(x::GPUArraysCore.AbstractGPUArray, y)
    reshape(Adapt.adapt(ArrayInterface.parameterless_type(x), y), Base.size(x)...)
end

function ArrayInterface.lu_instance(A::GPUArraysCore.AbstractGPUMatrix{T}) where {T}
    lu(Adapt.adapt(ArrayInterface.parameterless_type(A), ones(T, 0, 0)))
end

# Doesn't do much, but makes a gigantic change to the dependency chain.
# ArrayInterface.device(::Type{<:GPUArraysCore.AbstractGPUArray}) = ArrayInterface.GPU()

end
