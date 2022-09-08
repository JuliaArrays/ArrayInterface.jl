module ArrayInterfaceCUDA

using Adapt
using ArrayInterface, ArrayInterfaceGPUArrays
using CUDA
using CUDA.CUSOLVER

using LinearAlgebra

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    if VERSION >= v"1.8-0"
        LinearAlgebra.qr!(A)
    else
        CUDA.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
    end
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

end # module
