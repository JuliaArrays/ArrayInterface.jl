using ArrayInterface, Test

@test ArrayInterface.ismutable(rand(3))

using StaticArrays
@test ArrayInterface.ismutable(@SVector [1,2,3]) == false
@test ArrayInterface.ismutable(@MVector [1,2,3]) == true

using LinearAlgebra
D=Diagonal([1,2,3,4])
rowind,colind=findstructralnz(D)
@test [D[rowind[i],colind[i]] for i in 1:4]==[1,2,3,4]

Bu = Bidiagonal([1,2,3,4], [7,8,9], :U)
rowind,colind=findstructralnz(Bu)
@test [Bu[rowind[i],colind[i]] for i in 1:7]==[1,7,2,8,3,9,4]
Bl = Bidiagonal([1,2,3,4], [7,8,9], :L)
rowind,colind=findstructralnz(Bl)
@test [Bl[rowind[i],colind[i]] for i in 1:7]==[1,7,2,8,3,9,4]

Tri=Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])
rowind,colind=findstructralnz(Tri)
@test [Tri[rowind[i],colind[i]] for i in 1:10]==[1,2,3,4,4,5,6,1,2,3]
STri=SymTridiagonal([1,2,3,4],[5,6,7])
rowind,colind=findstructralnz(STri)
@test [STri[rowind[i],colind[i]] for i in 1:10]==[1,2,3,4,5,6,7,5,6,7]
