module ArrayInterfaceCUDA

using Adapt
using ArrayInterfaceCore
using CUDA

ArrayInterfaceCore.fast_scalar_indexing(::Type{<:CUDA.CuArray}) = false
@inline ArrayInterfaceCore.allowed_getindex(x::CUDA.CuArray, i...) = CUDA.@allowscalar(x[i...])
@inline ArrayInterfaceCore.allowed_setindex!(x::CUDA.CuArray, v, i...) = (CUDA.@allowscalar(x[i...] = v))

function Base.setindex(x::CUDA.CuArray, v, i::Int)
    _x = copy(x)
    ArrayInterfaceCore.allowed_setindex!(_x, v, i)
    return _x
end

function ArrayInterfaceCore.restructure(x::CUDA.CuArray, y)
    reshape(Adapt.adapt(ArrayInterfaceCore.parameterless_type(x), y), Base.size(x)...)
end

ArrayInterfaceCore.device(::Type{<:CUDA.CuArray}) = ArrayInterfaceCore.GPU()

function ArrayInterfaceCore.lu_instance(A::CuMatrix{T}) where {T}
    CUDA.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
end

end # module
