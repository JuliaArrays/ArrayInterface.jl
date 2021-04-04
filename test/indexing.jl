
#=
@btime ArrayInterface.argdims(ArrayInterface.DefaultArrayStyle(), $((1, CartesianIndex(1,2))))
  0.045 ns (0 allocations: 0 bytes)

@btime ArrayInterface.argdims(ArrayInterface.DefaultArrayStyle(), $((1, [CartesianIndex(1,2), CartesianIndex(1,3)])))
  0.047 ns (0 allocations: 0 bytes)
=#

@testset "argdims" begin
    @test @inferred(ArrayInterface.argdims(ArrayInterface.DefaultArrayStyle(), (1, CartesianIndex(1,2)))) === static((0, 2))
    @test @inferred(ArrayInterface.argdims(ArrayInterface.DefaultArrayStyle(), (1, [CartesianIndex(1,2), CartesianIndex(1,3)]))) === static((0, 2))
    @test @inferred(ArrayInterface.argdims(ArrayInterface.DefaultArrayStyle(), (1, CartesianIndex((2,2))))) === static((0, 2))
    @test @inferred(ArrayInterface.argdims(ArrayInterface.DefaultArrayStyle(), (CartesianIndex((2,2)), :, :))) === static((2, 1, 1))
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
    @test @inferred(ArrayInterface.to_index(axis, CartesianIndices(()))) === CartesianIndices(())

    x = LinearIndices((static(0):static(3),static(3):static(5),static(-2):static(0)));
    @test @inferred(ArrayInterface.to_index(x, (0, 3, -2))) === 1
    @test @inferred(ArrayInterface.to_index(x, (static(0), static(3), static(-2)))) === static(1)

    @test_throws BoundsError ArrayInterface.to_index(axis, 4)
    @test_throws BoundsError ArrayInterface.to_index(axis, 1:4)
    @test_throws BoundsError ArrayInterface.to_index(axis, [1, 2, 5])
    @test_throws BoundsError ArrayInterface.to_index(axis, [true, false, false, true])
    @test_throws ArgumentError ArrayInterface.to_index(axis, error)
end

@testset "unsafe_reconstruct" begin
    one_to = Base.OneTo(10)
    opt_ur = StaticInt(1):10
    ur = 1:10
    @test @inferred(ArrayInterface.unsafe_reconstruct(one_to, opt_ur)) === one_to
    @test @inferred(ArrayInterface.unsafe_reconstruct(one_to, one_to)) === one_to

    @test @inferred(ArrayInterface.unsafe_reconstruct(opt_ur, opt_ur)) === opt_ur
    @test @inferred(ArrayInterface.unsafe_reconstruct(opt_ur, one_to)) === opt_ur

    @test @inferred(ArrayInterface.unsafe_reconstruct(ur, ur)) === ur
    @test @inferred(ArrayInterface.unsafe_reconstruct(ur, one_to)) === ur
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
    @test @inferred(ArrayInterface.to_indices(a, ([CartesianIndex(1,1,1), CartesianIndex(1,2,1)],))) == (CartesianIndex{3}[CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1)],)
    @test @inferred(ArrayInterface.to_indices(a, ([CartesianIndex(1,1), CartesianIndex(1,2)],1:1))) == (CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(1, 2)], 1:1)
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
    @test @inferred(ArrayInterface.to_axes(A, (axis, axis), (multi_inds,))) === (static(1):2,)

    @test ArrayInterface.to_axis(axis, axis) === axis
    @test ArrayInterface.to_axis(axis, ArrayInterface.indices(axis)) === axis
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
    x = Array{Int,2}(undef, (2, 2))
    ArrayInterface.unsafe_set_element!(x, 1, (2, 2))
    @test ArrayInterface.unsafe_get_element(x, (2, 2)) === 1

    @test_throws MethodError ArrayInterface.unsafe_set_element!(x, 1, (:x, :x))
    @test_throws MethodError ArrayInterface.unsafe_get_element(x, (:x, :x))
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

    x = LinearIndices((static(0):static(3),static(3):static(5),static(-2):static(0)));
    @test @inferred(ArrayInterface.getindex(x, 0, 3, -2)) === 1
    @test @inferred(ArrayInterface.getindex(x, static(0), static(3), static(-2))) === 1

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

