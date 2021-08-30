
A = zeros(3, 4, 5);
A[:] = 1:60
Ap = @view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])';

ap_index = ArrayInterface.StrideIndex(Ap)
for x_i in axes(Ap, 1)
    for y_i in axes(Ap, 2)
        @test ap_index[x_i, y_i] == ap_index[x_i, y_i]
    end
end
@test @inferred(ArrayInterface.known_offsets(ap_index)) === ArrayInterface.known_offsets(Ap)
@test @inferred(ArrayInterface.known_offset1(ap_index)) === ArrayInterface.known_offset1(Ap)
@test @inferred(ArrayInterface.offsets(ap_index, 1)) === ArrayInterface.offset1(Ap)
@test @inferred(ArrayInterface.offsets(ap_index, static(1))) === ArrayInterface.offset1(Ap)
@test @inferred(ArrayInterface.known_strides(ap_index)) === ArrayInterface.known_strides(Ap)
@test @inferred(ArrayInterface.contiguous_axis(ap_index)) == 1
@test @inferred(ArrayInterface.contiguous_axis(ArrayInterface.StrideIndex{2,(1,2),Nothing,NTuple{2,Int},NTuple{2,Int}})) == nothing
@test @inferred(ArrayInterface.stride_rank(ap_index)) == (1, 3)

