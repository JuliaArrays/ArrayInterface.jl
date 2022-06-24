using StaticArrays, ArrayInterfaceCore, ArrayInterfaceStaticArraysCore, Test

x = @SVector [1,2,3]
@test ArrayInterfaceCore.ismutable(x) == false
@test ArrayInterfaceCore.ismutable(view(x, 1:2)) == false
@test ArrayInterfaceCore.can_setindex(typeof(x)) == false
@test ArrayInterfaceCore.buffer(x) == x.data

x = @MVector [1,2,3]
@test ArrayInterfaceCore.ismutable(x) == true
@test ArrayInterfaceCore.ismutable(view(x, 1:2)) == true

A = @SMatrix(randn(5, 5))
@test ArrayInterfaceCore.lu_instance(A) isa typeof(lu(A))
A = @MMatrix(randn(5, 5))
@test ArrayInterfaceCore.lu_instance(A) isa typeof(lu(A))

x = @SMatrix rand(Float32, 2, 2)
y = @SVector rand(4)
yr = ArrayInterfaceCore.restructure(x, y)
@test yr isa SMatrix{2, 2}
@test Base.size(yr) == (2,2)
@test vec(yr) == vec(y)
z = rand(4)
zr = ArrayInterfaceCore.restructure(x, z)
@test zr isa SMatrix{2, 2}
@test Base.size(zr) == (2,2)
@test vec(zr) == vec(z)