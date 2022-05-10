
using ArrayInterfaceCore
using ArrayInterfaceOffsetArrays
using OffsetArrays
using Static
using Test

A = zeros(3, 4, 5);
O = OffsetArray(A, 3, 7, 10);
Op = PermutedDimsArray(O,(3,1,2));
@test @inferred(ArrayInterfaceCore.offsets(O)) === (4, 8, 11)
@test @inferred(ArrayInterfaceCore.offsets(Op)) === (11, 4, 8)

@test @inferred(ArrayInterfaceCore.offsets((1,2,3))) === (StaticInt(1),)
o = OffsetArray(vec(A), 8);
@test @inferred(ArrayInterfaceCore.offset1(o)) === 9

@test @inferred(ArrayInterfaceCore.device(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173))) === ArrayInterfaceCore.CPUPointer()
@test @inferred(ArrayInterfaceCore.device(view(OffsetArray(A,2,3,-12), 4, :, -11:-9))) === ArrayInterfaceCore.CPUPointer()
@test @inferred(ArrayInterfaceCore.device(view(OffsetArray(A,2,3,-12), 3, :, [-11,-10,-9])')) === ArrayInterfaceCore.CPUIndex()

@test @inferred(ArrayInterfaceCore.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),1)) === Base.Slice(ArrayInterfaceCore.OptionallyStaticUnitRange(4,6))
@test @inferred(ArrayInterfaceCore.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),2)) === Base.Slice(ArrayInterfaceCore.OptionallyStaticUnitRange(-172,-170))

@test @inferred(ArrayInterfaceCore.device(OffsetArray(1:10))) === ArrayInterfaceCore.CPUIndex()
@test @inferred(ArrayInterfaceCore.device(OffsetArray(@view(reshape(1:8, 2,2,2)[1,1:2,:]),-3,4))) === ArrayInterfaceCore.CPUIndex()
@test @inferred(ArrayInterfaceCore.device(OffsetArray(zeros(2,2,2),8,-2,-5))) === ArrayInterfaceCore.CPUPointer()

