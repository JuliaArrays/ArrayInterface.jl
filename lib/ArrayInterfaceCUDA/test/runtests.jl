using ArrayInterfaceCUDA, CUDA, Test

A = cu(rand(4,4))
@test ArrayInterface.lu_instance(A) isa CUDA.CUSOLVER.CuQR
