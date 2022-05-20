module ArrayInterfaceCUDA

using Adapt
using ArrayInterface
using CUDA

const CanonicalInt = Union{Int,StaticInt}

ArrayInterface.fast_scalar_indexing(::Type{<:CUDA.CuArray}) = false
@inline ArrayInterface.allowed_getindex(x::CUDA.CuArray, i...) = CUDA.@allowscalar(x[i...])
@inline ArrayInterface.allowed_setindex!(x::CUDA.CuArray, v, i...) = (CUDA.@allowscalar(x[i...] = v))

function Base.setindex(x::CUDA.CuArray, v, i::Int)
    _x = copy(x)
    ArrayInterface.allowed_setindex!(_x, v, i)
    return _x
end

function ArrayInterface.restructure(x::CUDA.CuArray, y)
    reshape(Adapt.adapt(ArrayInterface.parameterless_type(x), y), Base.size(x)...)
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    CUDA.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
end

end # module
