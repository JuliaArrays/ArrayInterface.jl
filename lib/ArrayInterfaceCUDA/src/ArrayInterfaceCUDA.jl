module ArrayInterfaceCUDA

using Adapt
using ArrayInterface, ArrayInterfaceGPUArrays
using CUDA
using CUDA.CUSOLVER

using LinearAlgebra

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    if VERSION >= v"1.8-"
        LinearAlgebra.lu!(A)
    else
        LinearAlgebra.lu(A)
    end
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

end # module
