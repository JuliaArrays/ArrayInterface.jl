module ArrayInterfaceCuArrays

using Adapt
using ArrayInterfaceCore
using CuArrays

ArrayInterfaceCore.fast_scalar_indexing(::Type{<:CuArrays.CuArray}) = false
@inline ArrayInterfaceCore.allowed_getindex(x::CuArrays.CuArray, i...) = CuArrays.@allowscalar(x[i...])
@inline function ArrayInterfaceCore.allowed_setindex!(x::CuArrays.CuArray, v, i...)
    (CuArrays.@allowscalar(x[i...] = v))
end

function Base.setindex(x::CuArrays.CuArray, v, i::Int)
    _x = copy(x)
    allowed_setindex!(_x, v, i)
    return _x
end

function ArrayInterfaceCore.restructure(x::CuArrays.CuArray, y)
    reshape(Adapt.adapt(parameterless_type(x), y), Base.size(x)...)
end

ArrayInterfaceCore.device(::Type{<:CuArrays.CuArray}) = ArrayInterfaceCore.GPU()

function ArrayInterfaceCore.lu_instance(A::CuMatrix{T}) where {T}
    CuArrays.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
end


end # module
