
A = zeros(3, 4, 5);
A[:] = 1:60;
Aview = view(A, :, :, 1);
# FastContiguousSubArray
Aview = view(A, :, :, 1);
index = ArrayInterface.LinearStrideIndex(Aview)
shaped = ArrayInterface.ShapedIndex(Aview)
for i in eachindex(IndexLinear(), Aview)
    @test index[i] == Aview[i]
end
for i in eachindex(IndexCartesian(), Aview)
    @test index[shaped[i]] == Aview[i]
end
@test ArrayInterface.known_offset1(index) == ArrayInterface.known(ArrayInterface.offset1(index))

# FastSubArray
Aview = view(A, 2, :, 1);
index = ArrayInterface.LinearStrideIndex(Aview)
for i in eachindex(IndexLinear(), Aview)
    @test index[i] == Aview[i]
end

# SubArray
Aview = view(A, 2, :, 1);
index = ArrayInterface.SubIndex(Aview)
shaped = ArrayInterface.ShapedIndex(A)
for i in eachindex(Aview)
    @test shaped[index[i]] == Aview[i]
end

stride_index = ArrayInterface.StrideIndex(A)
Aperm = PermutedDimsArray(A,(3,1,2))
perm_index = ArrayInterface.PermutedIndex(Aperm)
Aview = view(Aperm, 2, 1:2, 1)
sub_index = ArrayInterface.SubIndex(Aview)
Aconj = Aview'
conj_index = ArrayInterface.ConjugateIndex()
multidim = ArrayInterface.MultidimIndex(Aconj)

composed = stride_index ∘ perm_index ∘ sub_index ∘ conj_index ∘ multidim
x1 = stride_index ∘ perm_index

for i in eachindex(IndexLinear(), Aconj)
    i0 = Aconj[i]
    i1 = multidim[i]
    i2 = conj_index[i1]
    i3 = sub_index[i2]
    i4 = perm_index[i3]
    i5 = stride_index[i4]
    @test Aconj[i1] == i0
    @test Aview[i2] == i0
    @test Aperm[i3] == i0
    @test A[i4] == i0
    @test i5 == i0
    @test composed[i] == i0
end

@test @inferred(ArrayInterface.known_offsets(stride_index)) === ArrayInterface.known_offsets(A)
@test @inferred(ArrayInterface.known_offset1(stride_index)) === ArrayInterface.known_offset1(A)
@test @inferred(ArrayInterface.known_strides(stride_index)) === ArrayInterface.known_strides(A)

A = zeros(3, 4, 5);
A[:] = 1:60;
Aview = view(A, :, 2, :);

Aview = view(A, 2, :, :);


