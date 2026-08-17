module ArrayInterfaceGPUArraysCoreExt

using Adapt
using ArrayInterface
import LinearAlgebra
import GPUArraysCore

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

# Build the `LU` directly rather than calling `lu` on an adapted array, matching the CPU
# methods in ArrayInterface.jl. Going through `lu` only works for backends that define their
# own, so it breaks on JLArrays (which downstream packages use to test GPU paths without a
# GPU) and on Metal, see #501 and #467.
function ArrayInterface.lu_instance(A::GPUArraysCore.AbstractGPUMatrix{T}) where {T}
    noUnitT = typeof(zero(T))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = similar(A, LinearAlgebra.BlasInt, 0)
    info = zero(LinearAlgebra.BlasInt)
    return LinearAlgebra.LU{luT}(similar(A, 0, 0), ipiv, info)
end

# Doesn't do much, but makes a gigantic change to the dependency chain.
# ArrayInterface.device(::Type{<:GPUArraysCore.AbstractGPUArray}) = ArrayInterface.GPU()

end
