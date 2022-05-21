module ArrayInterfaceCUDA

using Adapt
using ArrayInterface, ArrayInterfaceGPUArrays
using CUDA

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    CUDA.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

end # module
