
@testset "argdims" begin
    static_argdims(x) = Val(ArrayInterface.argdims(ArrayInterface.DefaultArrayStyle(), x))
    @test @inferred(static_argdims((1, CartesianIndex(1,2)))) === Val((0, 2))
    @test @inferred(static_argdims((1, [CartesianIndex(1,2), CartesianIndex(1,3)]))) === Val((0, 2))
    @test @inferred(static_argdims((1, CartesianIndex((2,2))))) === Val((0, 2))
    @test @inferred(static_argdims((CartesianIndex((2,2)), :, :))) === Val((2, 1, 1))
end

@testset "UnsafeIndex" begin
    @test @inferred(ArrayInterface.UnsafeIndex(ones(2,2,2), typeof((1,[1,2],1)))) == ArrayInterface.UnsafeGetCollection() 
    @test @inferred(ArrayInterface.UnsafeIndex(ones(2,2,2), typeof((1,1,1)))) == ArrayInterface.UnsafeGetElement() 
end

@testset "to_index" begin
    axis = 1:3
    @test @inferred(ArrayInterface.to_index(axis, 1)) === 1
    @test @inferred(ArrayInterface.to_index(axis, 1:2)) === 1:2
    @test @inferred(ArrayInterface.to_index(axis, [1, 2])) == [1, 2]
    @test @inferred(ArrayInterface.to_index(axis, [true, false, false])) == [1]

    @test_throws BoundsError  ArrayInterface.to_index(axis, 4)
    @test_throws BoundsError  ArrayInterface.to_index(axis, 1:4)
    @test_throws BoundsError ArrayInterface.to_index(axis, [1, 2, 5])
    @test_throws BoundsError  ArrayInterface.to_index(axis, [true, false, false, true])
end

@testset "to_indices" begin
    a = ones(2,2,1)
    v = ones(2)

    @testset "linear indexing" begin
        @test @inferred(ArrayInterface.to_indices(a, (1,))) == (1,)
        @test @inferred(ArrayInterface.to_indices(a, (1:2,))) == (1:2,)

        @testset "Linear indexing doesn't ruin vector indexing" begin
            @test @inferred(ArrayInterface.to_indices(v, (1:2,))) == (1:2,)
            @test @inferred(ArrayInterface.to_indices(v, (1,))) == (1,)
        end
    end

    @test @inferred(ArrayInterface.to_indices(ones(2,2,2), ([true,true], CartesianIndex(1,1)))) == ([1, 2], 1, 1)
    @test @inferred(ArrayInterface.to_indices(a, (1, 1))) == (1, 1)
    @test @inferred(ArrayInterface.to_indices(a, (1, 1:2))) == (1, 1:2)
    @test @inferred(ArrayInterface.to_indices(a, (1:2, 1))) == (1:2, 1)
    @test @inferred(ArrayInterface.to_indices(a, (1, :))) == (1, Base.Slice(1:2))
    @test @inferred(ArrayInterface.to_indices(a, (:, 1))) == (Base.Slice(1:2), 1)
    @test @inferred(ArrayInterface.to_indices(a, ([true, true], :))) == (Base.LogicalIndex(Bool[1, 1]), Base.Slice(1:2))
    @test @inferred(ArrayInterface.to_indices(a, (CartesianIndices((1,)), 1))) == (1:1, 1)
    @test @inferred(ArrayInterface.to_indices(a, (1, 1, 1))) == (1,1, 1)
    @test @inferred ArrayInterface.to_indices(a, ([CartesianIndex(1,1,1), CartesianIndex(1,2,1)],)) == (CartesianIndex{3}[CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1)],)
    @test @inferred ArrayInterface.to_indices(a, ([CartesianIndex(1,1), CartesianIndex(1,2)],1:1)) == (CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(1, 2)], 1:1)
    @test @inferred(first(ArrayInterface.to_indices(a, (fill(true, 2, 2, 1),)))) isa Base.LogicalIndex

    @test_throws BoundsError ArrayInterface.to_indices(a, (fill(true, 2, 2, 2),))
    @test_throws ErrorException ArrayInterface.to_indices(ones(2,2,2), (1, 1))
end

@testset "to_axes" begin
    A = ones(3, 3)
    axis = StaticInt(1):StaticInt(3)
    inds = StaticInt(1):StaticInt(2)
    multi_inds = [CartesianIndex(1, 1), CartesianIndex(1, 2)]

    @test @inferred(ArrayInterface.to_axes(A, (axis, axis), (inds, inds))) === (inds, inds)
    # vector indexing
    @test @inferred(ArrayInterface.to_axes(ones(3), (axis,), (inds,))) === (inds,)
    # linear indexing
    @test @inferred(ArrayInterface.to_axes(A, (axis, axis), (inds,))) === (inds,)
    # multidim arg
    @test @inferred(ArrayInterface.to_axes(A, (axis, axis), (multi_inds,))) === (Base.OneTo(2),)
end

@testset "0-dimensional" begin
    x = Array{Int,0}(undef)
    ArrayInterface.setindex!(x, 1)
    @test @inferred(ArrayInterface.getindex(x)) == 1
end

@testset "1-dimensional" begin
    for i = 1:3
        @test @inferred(ArrayInterface.getindex(LinearIndices((3,)), i)) == i
        @test @inferred(ArrayInterface.getindex(CartesianIndices((3,)), i)) == CartesianIndex(i,)
    end
    @test @inferred(ArrayInterface.getindex(LinearIndices((3,)), 2,1)) == 2
    @test @inferred(ArrayInterface.getindex(LinearIndices((3,)), [1])) == [1]
    # !!NOTE!! this is different than Base.getindex(::LinearIndices, ::AbstractUnitRange)
    # which returns a UnitRange. Instead we try to preserve axes if at all possible so the
    # values are the same but it's still wrapped in LinearIndices struct
    @test @inferred(ArrayInterface.getindex(LinearIndices((3,)), 1:2)) == 1:2
    @test @inferred(ArrayInterface.getindex(LinearIndices((3,)), 1:2:3)) === 1:2:3
    @test_throws BoundsError ArrayInterface.getindex(LinearIndices((3,)), 2:4)
    @test_throws BoundsError ArrayInterface.getindex(CartesianIndices((3,)), 2, 2)
    #   ambiguity btw cartesian indexing and linear indexing in 1d when
    #   indices may be nontraditional
    # TODO should this be implemented in ArrayInterface with vectorization?
    #@test_throws ArgumentError Base._sub2ind((1:3,), 2)
    #@test_throws ArgumentError Base._ind2sub((1:3,), 2)
end

@testset "2-dimensional" begin
    k = 0
    cartesian = CartesianIndices((4,3))
    linear = LinearIndices(cartesian)
    for j = 1:3, i = 1:4
        k += 1
        @test @inferred(ArrayInterface.getindex(linear, i, j)) == ArrayInterface.getindex(linear, k) == k
        @test @inferred(ArrayInterface.getindex(cartesian, k)) == CartesianIndex(i,j)
        @test @inferred(ArrayInterface.getindex(LinearIndices(map(Base.Slice, (0:3,3:5))), i-1, j+2)) == k
        @test @inferred(ArrayInterface.getindex(CartesianIndices(map(Base.Slice, (0:3,3:5))), k)) == CartesianIndex(i-1,j+2)
    end
    @test @inferred(ArrayInterface.getindex(linear, linear)) == linear
    @test @inferred(ArrayInterface.getindex(linear, vec(linear))) == vec(linear)
    @test @inferred(ArrayInterface.getindex(linear, cartesian)) == linear
    @test @inferred(ArrayInterface.getindex(linear, vec(cartesian))) == vec(linear)
    @test @inferred(ArrayInterface.getindex(cartesian, linear)) == cartesian
    @test @inferred(ArrayInterface.getindex(cartesian, vec(linear))) == vec(cartesian)
    @test @inferred(ArrayInterface.getindex(cartesian, cartesian)) == cartesian
    @test @inferred(ArrayInterface.getindex(cartesian, vec(cartesian))) == vec(cartesian)
    @test @inferred(ArrayInterface.getindex(linear, 2:3)) === 2:3
    @test @inferred(ArrayInterface.getindex(linear, 3:-1:1)) === 3:-1:1
    @test_throws BoundsError ArrayInterface.getindex(linear, 4:13)
end

@testset "3-dimensional" begin
    l = 0
    for k = 1:2, j = 1:3, i = 1:4
        l += 1
        @test @inferred(ArrayInterface.getindex(LinearIndices((4,3,2)),i,j,k)) == l
        @test @inferred(ArrayInterface.getindex(LinearIndices((4,3,2)),l)) == l
        @test @inferred(ArrayInterface.getindex(CartesianIndices((4,3,2)),i,j,k)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterface.getindex(CartesianIndices((4,3,2)),l)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterface.getindex(LinearIndices((1:4,1:3,1:2)),i,j,k)) == l
        @test @inferred(ArrayInterface.getindex(LinearIndices((1:4,1:3,1:2)),l)) == l
        @test @inferred(ArrayInterface.getindex(CartesianIndices((1:4,1:3,1:2)),i,j,k)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterface.getindex(CartesianIndices((1:4,1:3,1:2)),l)) == CartesianIndex(i,j,k)
    end

    l = 0
    for k = -101:-100, j = 3:5, i = 0:3
        l += 1
        @test @inferred(ArrayInterface.getindex(LinearIndices(map(Base.Slice, (0:3,3:5,-101:-100))),i,j,k)) == l
        @test @inferred(ArrayInterface.getindex(LinearIndices(map(Base.Slice, (0:3,3:5,-101:-100))),l)) == l
        @test @inferred(ArrayInterface.getindex(CartesianIndices(map(Base.Slice, (0:3,3:5,-101:-100))),i,j,k)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterface.getindex(CartesianIndices(map(Base.Slice, (0:3,3:5,-101:-100))),l)) == CartesianIndex(i,j,k)
    end

    local A = reshape(Vector(1:9), (3,3))
    @test @inferred(ArrayInterface.getindex(CartesianIndices(size(A)),6)) == CartesianIndex(3,2)
    @test @inferred(ArrayInterface.getindex(LinearIndices(size(A)),3, 2)) == 6
    @test @inferred(ArrayInterface.getindex(CartesianIndices(A),6)) == CartesianIndex(3,2)
    @test @inferred(ArrayInterface.getindex(LinearIndices(A),3, 2)) == 6
    for i in 1:length(A)
        @test @inferred(ArrayInterface.getindex(LinearIndices(A),ArrayInterface.getindex(CartesianIndices(A),i))) == i
    end
end

@testset "IndexMap" begin
    x = ones(Int, 4, 4,4,4);
    inds = (Base.Slice(1:4),Base.Slice(1:4),1:4,Base.Slice(1:4))
    ArrayInterface.index_by_pointer(ones(Int, 4, 4,4,4), (Base.Slice(1:4),Base.Slice(1:4),1:4,Base.Slice(1:4)))

    loop = ArrayInterface.index_loops(x, inds)
    index_expr(index_loops(x, inds), :inds, typeof(inds), :s, typeof(s), :f, typeof(f))

    @test ArrayInterface.index_maps(x, inds) == (IndexMap((1,2),(1,2)),IndexMap(3,3),IndexMap(4,4))
    inds = (Base.Slice(1:4),Base.Slice(1:4),1,Base.Slice(1:4))
    @test ArrayInterface.index_maps(x, inds) == (IndexMap((1,2),(1,2)),IndexMap(3,nothing),IndexMap(4,3))
    inds = (3:4, 3, 4,5)
    @test ArrayInterface.index_maps(x, inds) == (IndexMap(1,1),IndexMap(2,nothing),IndexMap(3,nothing),IndexMap(4,nothing))

    ArrayInterface.combine_index_maps(ArrayInterface.index_maps(x, inds))
    ArrayInterface.index_loops(x, inds)
end



#=
inds = (One():StaticInt(4), One():StaticInt(4), One():StaticInt(4))
strs = (One(), StaticInt(4), StaticInt(16))
itr = StridedIndexingIterator(inds, strs, One())
[i for i in itr]

=#

A = zeros(Int, 3,4,5, 6);
A[LinearIndices(A)] .= LinearIndices(A);
Ap = PermutedDimsArray(A, (4,1,2, 3));
Apv = @view(Ap[2,:,2:3])
Apvadj = Apv'

A = zeros(Int, 3,4,5);
A[LinearIndices(A)] .= LinearIndices(A);
Ap = PermutedDimsArray(A, (3,1,2));
Apv = @view(Ap[2,:,2:3])
Apvadj = Apv'

collect(StridedIterator(A)) == [A...]
collect(StridedIndexingIterator(Ap)) == [Ap...]
collect(StridedIndexingIterator(Apv)) == [Apv...]
collect(StridedIndexingIterator(Apvadj)) == [Apvadj...]

function test_pointer_iterator(x)
    src = pointer(x)
    for (i, x_i) in zip(StridedIndexingIterator(A),x)
        @test unsafe_load(src, i)
    end
end


using ArrayInterface: IndexMap

collect(StridedIterator((1:4, 2:4), (1, 1), (1, 4), 1))

A = ones(Int, 4, 4);
A[LinearIndices(A)] .= LinearIndices(A);
Ap = PermutedDimsArray(A, (2,1,3,4));

@inferred ArrayInterface.index_loops(A, (1:4, 2))
A[Base.Slice(axes(A, 1)), 2]


ArrayInterface.index_by_pointer(A, (Base.Slice(axes(A, 1)), Base.Slice(axes(A, 2))))
ArrayInterface.index_by_pointer(A, (Base.Slice(axes(A, 1)), Base.Slice(axes(A, 2))))
         , 1, 1)

ArrayInterface.index_by_pointer(A, (1:2, 1:2, 1, 2))

indexing_map(A, inds4)

ArrayInterface.contiguous_axis_indicator(Ap)
ArrayInterface.contiguous_axis_indicator(view(Ap, 1, :, :,1))
ArrayInterface.contiguous_axis_indicator(view(A, :, :, :, 3))

itr = StridedIterator(A)
Av = @view(A[:, 2:4])

collect(StridedIterator(Av))

A = ones(Int, 4, 4,4,4);
indexing_map(ones(Int, 4, 4,4,4), (1,1:4,1:4,1:4))

src = pointer(A)
dst = ArrayInterface.allocate_memory(A, 4)

unsafe_copyto!(dst + m[1][1], src + m[1][2], 1)
unsafe_copyto!(dst + m[2][1], src + m[2][2], 1)
unsafe_copyto!(dst + m[3][1], src + m[3][2], 1)
unsafe_copyto!(dst + m[4][1], src + m[4][2], 1)
unsafe_wrap(Array, dst, (4,))


unsafe_load(src, 1)
unsafe_store!(dst, unsafe_load(src, 4), 2)
unsafe_load(dst, 2)

@btime ArrayInterface.unsafe_get_collection(A, (Base.Slice(axes(A, 1)), 2))

