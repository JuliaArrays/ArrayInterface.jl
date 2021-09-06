
function test_array_index(x)
    @testset "$x" begin
        linear_idx = @inferred(ArrayInterface.ArrayIndex{1}(x))
        b = ArrayInterface.buffer(x)
        for i in eachindex(IndexLinear(), x)
            @test b[linear_idx[i]] == x[i]
        end
        cartesian_idx = @inferred(ArrayInterface.ArrayIndex{ndims(x)}(x))
        for i in eachindex(IndexCartesian(), x)
            @test b[cartesian_idx[i]] == x[i]
        end
    end
end

A = zeros(3, 4, 5);
A[:] = 1:60;
Aperm = PermutedDimsArray(A,(3,1,2));
Aview = @view(Aperm[:,1:2,1]);
Ap = Aview';
Apperm = PermutedDimsArray(Ap, (2, 1));

test_array_index(A)
test_array_index(Aperm)
test_array_index(Aview)
test_array_index(Ap)
test_array_index(view(A, :, :, 1))  # FastContiguousSubArray
test_array_index(view(A, 2, :, :))  # FastSubArray

idx = @inferred(ArrayInterface.ArrayIndex{3}(A)[ArrayInterface.ArrayIndex{3}(Aperm)])
for i in eachindex(IndexCartesian(), Aperm)
    @test A[idx[i]] == Aperm[i]
end
idx = @inferred(idx[ArrayInterface.ArrayIndex{2}(Aview)])
for i in eachindex(IndexCartesian(), Aview)
    @test A[idx[i]] == Aview[i]
end
idx = @inferred(idx[ArrayInterface.ArrayIndex{2}(Ap)])
for i in eachindex(IndexCartesian(), Ap)
    @test A[idx[i]] == Ap[i]
end
idx = @inferred(idx[ArrayInterface.ArrayIndex{2}(Apperm)])
for i in eachindex(IndexCartesian(), Apperm)
    @test A[idx[i]] == Apperm[i]
end

idx = @inferred(ArrayInterface.ArrayIndex{1}(1:2))
@test idx[@inferred(ArrayInterface.ArrayIndex{1}((1:2)'))] isa ArrayInterface.OffsetIndex{StaticInt{0}}
@test @inferred(ArrayInterface.ArrayIndex{2}((1:2)'))[CartesianIndex(1, 2)] == 2
@test @inferred(ArrayInterface.ArrayIndex{1}(1:2)) isa ArrayInterface.OffsetIndex{StaticInt{0}}
@test @inferred(ArrayInterface.ArrayIndex{1}((1:2)')) isa ArrayInterface.OffsetIndex{StaticInt{0}}
@test @inferred(ArrayInterface.ArrayIndex{1}(PermutedDimsArray(1:2, (1,)))) isa ArrayInterface.OffsetIndex{StaticInt{0}}
@test @inferred(ArrayInterface.ArrayIndex{1}(reshape(1:10, 2, 5))) isa ArrayInterface.OffsetIndex{StaticInt{0}}
@test @inferred(ArrayInterface.ArrayIndex{2}(reshape(1:10, 2, 5))) isa ArrayInterface.StrideIndex

ap_index = ArrayInterface.StrideIndex(Ap)
@test @inferred(ArrayInterface.known_offsets(ap_index)) === ArrayInterface.known_offsets(Ap)
@test @inferred(ArrayInterface.known_offset1(ap_index)) === ArrayInterface.known_offset1(Ap)
@test @inferred(ArrayInterface.offsets(ap_index, 1)) === ArrayInterface.offset1(Ap)
@test @inferred(ArrayInterface.offsets(ap_index, static(1))) === ArrayInterface.offset1(Ap)
@test @inferred(ArrayInterface.known_strides(ap_index)) === ArrayInterface.known_strides(Ap)
@test @inferred(ArrayInterface.contiguous_axis(ap_index)) == 1
@test @inferred(ArrayInterface.contiguous_axis(ArrayInterface.StrideIndex{2,(1,2),nothing,NTuple{2,Int},NTuple{2,Int}})) == nothing
@test @inferred(ArrayInterface.stride_rank(ap_index)) == (1, 3)

