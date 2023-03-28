using StaticArrays, ArrayInterface, Test
using LinearAlgebra
using ArrayInterface: undefmatrix, zeromatrix

x = @SVector [1,2,3]
@test ArrayInterface.ismutable(x) == false
@test ArrayInterface.ismutable(view(x, 1:2)) == false
@test ArrayInterface.can_setindex(typeof(x)) == false
@test ArrayInterface.buffer(x) == x.data

x = @MVector [1,2,3]
@test ArrayInterface.ismutable(x) == true
@test ArrayInterface.ismutable(view(x, 1:2)) == true

A = @SMatrix(randn(5, 5))
@test ArrayInterface.lu_instance(A) isa typeof(lu(A))
A = @MMatrix(randn(5, 5))
@test ArrayInterface.lu_instance(A) isa typeof(lu(A))

x = @SMatrix rand(Float32, 2, 2)
y = @SVector rand(4)
yr = ArrayInterface.restructure(x, y)
@test yr isa SMatrix{2, 2}
@test Base.size(yr) == (2,2)
@test vec(yr) == vec(y)
z = rand(4)
zr = ArrayInterface.restructure(x, z)
@test zr isa SMatrix{2, 2}
@test Base.size(zr) == (2,2)
@test vec(zr) == vec(z)


@testset "zeromatrix and unsafematrix" begin
    for T in (Int, Float32, Float64)
        for (vectype, mattype) in ((SVector{4,T},     SMatrix{4,4,T,16}),
                                   (MVector{4,T},     MMatrix{4,4,T,16}),
                                   (SMatrix{2,2,T,4}, SMatrix{4,4,T,16}),
                                   (MMatrix{2,2,T,4}, MMatrix{4,4,T,16}))
            v = vectype(rand(T, 4))
            um = undefmatrix(v)
            @test typeof(um) == mattype
            @test zeromatrix(v) == zeros(T,length(v),length(v))
        end
    end
end
