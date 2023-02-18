using ArrayInterface
using OffsetArrays
using Static
using Test

A = zeros(3, 4, 5);
O = OffsetArray(A, 3, 7, 10);
Op = PermutedDimsArray(O,(3,1,2));
@test @inferred(ArrayInterface.offsets(O)) === (4, 8, 11)
@test @inferred(ArrayInterface.offsets(Op)) === (11, 4, 8)

@test @inferred(ArrayInterface.static_to_indices(O, (:, :, :))) == (4:6, 8:11, 11:15)
@test @inferred(ArrayInterface.static_to_indices(Op, (:, :, :))) == (11:15, 4:6, 8:11)

@test @inferred(ArrayInterface.offsets((1,2,3))) === (StaticInt(1),)
o = OffsetArray(vec(A), 8);
@test @inferred(ArrayInterface.offset1(o)) === 9

@test @inferred(ArrayInterface.device(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173))) === ArrayInterface.CPUPointer()
@test @inferred(ArrayInterface.device(view(OffsetArray(A,2,3,-12), 4, :, -11:-9))) === ArrayInterface.CPUPointer()
@test @inferred(ArrayInterface.device(view(OffsetArray(A,2,3,-12), 3, :, [-11,-10,-9])')) === ArrayInterface.CPUIndex()

@test @inferred(ArrayInterface.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),1)) === Base.Slice(Static.OptionallyStaticUnitRange(4,6))
@test @inferred(ArrayInterface.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),2)) === Base.Slice(Static.OptionallyStaticUnitRange(-172,-170))

@test @inferred(ArrayInterface.device(OffsetArray(1:10))) === ArrayInterface.CPUIndex()
@test @inferred(ArrayInterface.device(OffsetArray(@view(reshape(1:8, 2,2,2)[1,1:2,:]),-3,4))) === ArrayInterface.CPUIndex()
@test @inferred(ArrayInterface.device(OffsetArray(zeros(2,2,2),8,-2,-5))) === ArrayInterface.CPUPointer()

offset_view = @view OffsetArrays.centered(zeros(eltype(A), 5, 5))[:, begin]; # SubArray of OffsetArray
@test @inferred(ArrayInterface.offsets(offset_view)) == (-2,)

B = OffsetArray(PermutedDimsArray(rand(2,3,4), (2,3,1)));
@test @inferred(ArrayInterface.StrideIndex(B)) === ArrayInterface.StrideIndex{3, (2, 3, 1), 3}((2, 6, static(1)), (1, 1, 1))
