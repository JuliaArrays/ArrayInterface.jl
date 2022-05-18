
A = zeros(3, 4, 5);
A[:] = 1:60
Ap = @view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])';

ap_index = ArrayInterfaceCore.StrideIndex(Ap)
for x_i in axes(Ap, 1)
    for y_i in axes(Ap, 2)
        @test ap_index[x_i, y_i] == ap_index[x_i, y_i]
    end
end
@test @inferred(ArrayInterfaceCore.known_offsets(ap_index)) === ArrayInterfaceCore.known_offsets(Ap)
@test @inferred(ArrayInterfaceCore.known_offset1(ap_index)) === ArrayInterfaceCore.known_offset1(Ap)
@test @inferred(ArrayInterfaceCore.offsets(ap_index, 1)) === ArrayInterfaceCore.offset1(Ap)
@test @inferred(ArrayInterfaceCore.offsets(ap_index, static(1))) === ArrayInterfaceCore.offset1(Ap)
@test @inferred(ArrayInterfaceCore.known_strides(ap_index)) === ArrayInterfaceCore.known_strides(Ap)
@test @inferred(ArrayInterfaceCore.contiguous_axis(ap_index)) == 1
@test @inferred(ArrayInterfaceCore.contiguous_axis(ArrayInterfaceCore.StrideIndex{2,(1,2),nothing,NTuple{2,Int},NTuple{2,Int}})) === nothing
@test @inferred(ArrayInterfaceCore.stride_rank(ap_index)) == (1, 3)

let v = Float64.(1:10)', v2 = transpose(parent(v))
  sv = @view(v[1:5])'
  sv2 = @view(v2[1:5])'
  @test @inferred(ArrayInterfaceCore.StrideIndex(sv)) === @inferred(ArrayInterfaceCore.StrideIndex(sv2)) === ArrayInterfaceCore.StrideIndex{2, (2, 1), 2}((StaticInt(1), StaticInt(1)), (StaticInt(1), StaticInt(1)))
  @test @inferred(ArrayInterfaceCore.stride_rank(parent(sv))) === @inferred(ArrayInterfaceCore.stride_rank(parent(sv2))) === (StaticInt(1),)
end

D=Diagonal([1,2,3,4])
@test has_sparsestruct(D)
rowind,colind=findstructralnz(D)
@test [D[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4]
@test length(rowind)==4
@test length(rowind)==length(colind)

Bu = Bidiagonal([1,2,3,4], [7,8,9], :U)
@test has_sparsestruct(Bu)
rowind,colind=findstructralnz(Bu)
@test [Bu[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,7,2,8,3,9,4]
Bl = Bidiagonal([1,2,3,4], [7,8,9], :L)
@test has_sparsestruct(Bl)
rowind,colind=findstructralnz(Bl)
@test [Bl[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,7,2,8,3,9,4]

Tri=Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])
@test has_sparsestruct(Tri)
rowind,colind=findstructralnz(Tri)
@test [Tri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4,4,5,6,1,2,3]

STri=SymTridiagonal([1,2,3,4],[5,6,7])
@test has_sparsestruct(STri)
rowind,colind=findstructralnz(STri)
@test [STri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4,5,6,7,5,6,7]

