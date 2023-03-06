module ArrayInterfaceCUDAExt

using ArrayInterface

if isdefined(Base, :get_extension)
    using CUDA
    using CUDA.CUSOLVER
    using LinearAlgebra
else
    using ..CUDA
    using ..CUDA.CUSOLVER
    using ..LinearAlgebra
end

function ArrayInterface.lu_instance(A::CuMatrix{T}) where {T}
    if VERSION >= v"1.8-"
        ipiv = cu(Vector{Int32}(undef, 0))
        info = zero(Int)
        return LinearAlgebra.LU(similar(A, 0, 0), ipiv, info)
    else
        LinearAlgebra.lu(A; check = false)
    end
end

ArrayInterface.device(::Type{<:CUDA.CuArray}) = ArrayInterface.GPU()

end # module
