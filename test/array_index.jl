

function test_array_index(x)
    @testset "$x" begin
        linear_idx = ArrayInterface.ArrayIndex{1}(x)
        b = ArrayInterface.buffer(x)
        for i in eachindex(IndexLinear(), x)
            @test b[linear_idx[i]] == x[i]
        end
        cartesian_idx = ArrayInterface.ArrayIndex{ndims(x)}(x)
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

#ArrayInterface.ArrayIndex{1}(x)

test_array_index(A)
test_array_index(Aperm)
test_array_index(Aview)
test_array_index(Ap)
test_array_index(view(A, :, :, 1))  # FastContiguousSubArray
test_array_index(view(A, 2, :, :))  # FastSubArray

ap_index = ArrayInterface.StrideIndex(Ap)
@test @inferred(ArrayInterface.known_offsets(ap_index)) === ArrayInterface.known_offsets(Ap)
@test @inferred(ArrayInterface.known_offset1(ap_index)) === ArrayInterface.known_offset1(Ap)
@test @inferred(ArrayInterface.offsets(ap_index, 1)) === ArrayInterface.offset1(Ap)
@test @inferred(ArrayInterface.offsets(ap_index, static(1))) === ArrayInterface.offset1(Ap)
@test @inferred(ArrayInterface.known_strides(ap_index)) === ArrayInterface.known_strides(Ap)
@test @inferred(ArrayInterface.contiguous_axis(ap_index)) == 1
@test @inferred(ArrayInterface.contiguous_axis(ArrayInterface.StrideIndex{2,(1,2),nothing,NTuple{2,Int},NTuple{2,Int}})) == nothing
@test @inferred(ArrayInterface.stride_rank(ap_index)) == (1, 3)


#=
using Revise
using Pkg
Pkg.activate(".")
using ArrayInterface
using ArrayInterface: buffer, array_index, LinearAccess, CartesianAccess

function test_layouts(x)
    index = ArrayInterface.array_index(x, LinearAccess())
    for i in eachindex(IndexLinear(), x)
        @test buffer(x)[index[i]] == x[i]
    end
    index = ArrayInterface.array_index(x, CartesianAccess())
    for i in eachindex(IndexCartesian(), x)
        @test buffer(x)[index[i]] == x[i]
    end

    lyt = ArrayInterface.layout(x, LinearAccess())
    for i in eachindex(IndexLinear(), x)
        @test lyt[i] == x[i]
    end

    lyt = ArrayInterface.layout(x, CartesianAccess())
    for i in eachindex(IndexCartesian(), x)
        @test lyt[i] == x[i]
    end
end

A = zeros(Int, 3, 4, 5);
A[:] = 1:60;
Aperm = PermutedDimsArray(A, (3,1,2));
Asub = @view(Aperm[:,1:2,1]);
Ap = Asub';

test_layouts(A)
test_layouts(Aperm)
test_layouts(Asub)
test_layouts(Ap)
test_layouts(view(A, :, :, 1))  # FastContiguousSubArray
test_layouts(view(A, 2, :, 1))  # FastSubArray



lyt = ArrayInterface.layout(view(A, 2, :, 1), LinearAccess())
for i in eachindex(IndexLinear(), x)
    @test lyt[i] == x[i]
end

function base_add(x)
    out = zero(eltype(x))
    @inbounds for i in eachindex(IndexCartesian(), x)
        out += x[i]
    end
    return out
end

function layout_add(x)
    out = zero(eltype(x))
    lyt = ArrayInterface.layout(x, ArrayInterface.CartesianAccess())
    @inbounds for i in eachindex(IndexCartesian(), x)
        out += lyt[i]
    end
    return out
end


@btime base_add($Ap)

@btime layout_add($Ap)


#=

lyt = ArrayInterface.layout(A, CartesianAccess())
lyt = ArrayInterface.layout(Ap, CartesianAccess())

lyt = ArrayInterface.layout(Ap, LinearAccess())
for i in eachindex(IndexCartesian(), Ap)
    @test lyt[i] == Ap[i]
end

@testset "FastContiguousSubArray" begin
    test_array_index(view(A, :, :, 1))
end
@testset "FastSubArray" begin
    test_array_index(view(A, 2, :, 1))
end

A = zeros(Int, 3, 4, 5);
A[:] = 1:60;
Aperm = PermutedDimsArray(A, (3,1,2));
Asub = @view(Aperm[:,1:2,1]);
Ap = Asub';

test_layout(Asub)

i1 = ArrayInterface.array_index(Aperm, LinearAccess())
i2 = ArrayInterface.array_index(parent(Aperm), CartesianAccess())
i1[i2]

i = ArrayInterface.array_index(Aperm, LinearAccess())
lyt[i]
    lyt = ArrayInterface.layout(Aperm, CartesianAccess())
    for i in eachindex(IndexCartesian(), Aperm)
        @test lyt[i] == Aperm[i]
    end

lyt = ArrayInterface.layout(A, CartesianAccess())
lyt = ArrayInterface.layout(Aperm, CartesianAccess())
lyt = ArrayInterface.layout(Asub, CartesianAccess())
lyt = ArrayInterface.layout(Ap, CartesianAccess())

test_array_index(A)
test_array_index(Aperm)
test_array_index(Asub)
test_array_index(Ap)

test_layout(A)
test_layout(Aperm)
test_layout(Asub)
test_layout(Ap)
=#

#=
Asub = view(A, 2, :, 1);
index = array_index(Asub, LinearAccess())
=#

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
=#
