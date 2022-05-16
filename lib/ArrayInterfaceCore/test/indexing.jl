
@testset "to_index" begin
    axis = 1:3
    @test @inferred(ArrayInterfaceCore.to_index(axis, 1)) === 1
    @test @inferred(ArrayInterfaceCore.to_index(axis, static(1))) === static(1)
    @test @inferred(ArrayInterfaceCore.to_index(axis, CartesianIndex(1))) === (1,)
    @test @inferred(ArrayInterfaceCore.to_index(axis, 1:2)) === 1:2
    @test @inferred(ArrayInterfaceCore.to_index(axis, CartesianIndices((1:2,)))) == (1:2,)
    @test @inferred(ArrayInterfaceCore.to_index(axis, [1, 2])) == [1, 2]
    @test @inferred(ArrayInterfaceCore.to_index(axis, [true, false, false])) == [1]
    index = @inferred(ArrayInterfaceCore.to_index(axis, :))
    @test @inferred(ArrayInterfaceCore.to_index(axis, index)) == index == ArrayInterfaceCore.indices(axis)

    x = LinearIndices((static(0):static(3),static(3):static(5),static(-2):static(0)));
    @test_throws ArgumentError ArrayInterfaceCore.to_index(axis, error)
end

@testset "unsafe_reconstruct" begin
    one_to = Base.OneTo(10)
    opt_ur = StaticInt(1):10
    ur = 1:10
    @test @inferred(ArrayInterfaceCore.unsafe_reconstruct(one_to, opt_ur)) === one_to
    @test @inferred(ArrayInterfaceCore.unsafe_reconstruct(one_to, one_to)) === one_to

    @test @inferred(ArrayInterfaceCore.unsafe_reconstruct(opt_ur, opt_ur)) === opt_ur
    @test @inferred(ArrayInterfaceCore.unsafe_reconstruct(opt_ur, one_to)) === opt_ur

    @test @inferred(ArrayInterfaceCore.unsafe_reconstruct(ur, ur)) === ur
    @test @inferred(ArrayInterfaceCore.unsafe_reconstruct(ur, one_to)) === ur
end

@testset "to_indices" begin
    a = ones(2,2,1)
    v = ones(2)

    @testset "linear indexing" begin
        @test @inferred(ArrayInterfaceCore.to_indices(a, (1,))) == (1,)
        @test @inferred(ArrayInterfaceCore.to_indices(a, (1:2,))) == (1:2,)

        @testset "Linear indexing doesn't ruin vector indexing" begin
            @test @inferred(ArrayInterfaceCore.to_indices(v, (1:2,))) == (1:2,)
            @test @inferred(ArrayInterfaceCore.to_indices(v, (1,))) == (1,)
        end
    end

    @test @inferred(ArrayInterfaceCore.to_indices(a, (CartesianIndices(()),))) == (CartesianIndices(()),)
    inds = @inferred(ArrayInterfaceCore.to_indices(a, (:,)))
    @test @inferred(ArrayInterfaceCore.to_indices(a, inds)) == inds == (ArrayInterfaceCore.indices(a),)
    @test @inferred(ArrayInterfaceCore.to_indices(ones(2,2,2), ([true,true], CartesianIndex(1,1)))) == ([1, 2], 1, 1)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1, 1))) == (1, 1)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1, CartesianIndex(1)))) == (1, 1)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1, false))) == (1, 0)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1, 1:2))) == (1, 1:2)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1:2, 1))) == (1:2, 1)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1, :))) == (1, Base.Slice(1:2))
    @test @inferred(ArrayInterfaceCore.to_indices(a, (:, 1))) == (Base.Slice(1:2), 1)
    @test @inferred(ArrayInterfaceCore.to_indices(a, ([true, true], :))) == (Base.LogicalIndex(Bool[1, 1]), Base.Slice(1:2))
    @test @inferred(ArrayInterfaceCore.to_indices(a, (CartesianIndices((1,)), 1))) == (1:1, 1)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1, 1, 1))) == (1,1, 1)
    @test @inferred(ArrayInterfaceCore.to_indices(a, (1, fill(true, 2, 2)))) == Base.to_indices(a, (1, fill(true, 2, 2)))
    inds = @inferred(ArrayInterfaceCore.to_indices(a, (fill(true, 2, 2, 1),)))
    # Conversion to LogicalIndex doesn't change
    @test @inferred(ArrayInterfaceCore.to_indices(a, inds)) == inds
    @test @inferred(ArrayInterfaceCore.to_indices(a, (fill(true, 2, 1), 1))) == Base.to_indices(a, (fill(true, 2, 1), 1))
    @test @inferred(ArrayInterfaceCore.to_indices(a, (fill(true, 2), 1, 1))) == Base.to_indices(a, (fill(true, 2), 1, 1))
    @test @inferred(ArrayInterfaceCore.to_indices(a, ([CartesianIndex(1,1,1), CartesianIndex(1,2,1)],))) == (CartesianIndex{3}[CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1)],)
    @test @inferred(ArrayInterfaceCore.to_indices(a, ([CartesianIndex(1,1), CartesianIndex(1,2)],1:1))) == (CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(1, 2)], 1:1)
    @test @inferred(first(ArrayInterfaceCore.to_indices(a, (fill(true, 2, 2, 1),)))) isa Base.LogicalIndex
end

@testset "splat indexing" begin
    struct SplatFirst end

    ArrayInterfaceCore.to_index(x, ::SplatFirst) = map(first, axes(x))
    ArrayInterfaceCore.is_splat_index(::Type{SplatFirst}) = static(true)
    x = rand(4,4,4,4,4,4,4,4,4,4)
    i = (1, SplatFirst(), 2, SplatFirst(), CartesianIndex(1, 1))

    @test @inferred(ArrayInterfaceCore.to_indices(x, i)) == (1, 1, 1, 1, 1, 1, 2, 1, 1, 1)
end

@testset "to_axes" begin
    A = ones(3, 3)
    axis = StaticInt(1):StaticInt(3)
    inds = StaticInt(1):StaticInt(2)
    multi_inds = [CartesianIndex(1, 1), CartesianIndex(1, 2)]

    @test @inferred(ArrayInterfaceCore.to_axes(A, (axis, axis), (inds, inds))) === (inds, inds)
    # vector indexing
    @test @inferred(ArrayInterfaceCore.to_axes(ones(3), (axis,), (inds,))) === (inds,)
    # linear indexing
    @test @inferred(ArrayInterfaceCore.to_axes(A, (axis, axis), (inds,))) === (inds,)
    # multidim arg
    @test @inferred(ArrayInterfaceCore.to_axes(A, (axis, axis), (multi_inds,))) === (static(1):2,)

    @test ArrayInterfaceCore.to_axis(axis, axis) === axis
    @test ArrayInterfaceCore.to_axis(axis, ArrayInterfaceCore.indices(axis)) === axis
end

@testset "0-dimensional" begin
    x = Array{Int,0}(undef)
    ArrayInterfaceCore.setindex!(x, 1)
    @test @inferred(ArrayInterfaceCore.getindex(x)) == 1
end

@testset "1-dimensional" begin
    for i = 1:3
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((3,)), i)) == i
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices((3,)), i)) == CartesianIndex(i,)
    end
    @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((3,)), 2,1)) == 2
    @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((3,)), [1])) == [1]
    # !!NOTE!! this is different than Base.getindex(::LinearIndices, ::AbstractUnitRange)
    # which returns a UnitRange. Instead we try to preserve axes if at all possible so the
    # values are the same but it's still wrapped in LinearIndices struct
    @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((3,)), 1:2)) == 1:2
    @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((3,)), 1:2:3)) === 1:2:3
    @test_throws BoundsError ArrayInterfaceCore.getindex(LinearIndices((3,)), 2:4)
    @test_throws BoundsError ArrayInterfaceCore.getindex(CartesianIndices((3,)), 2, 2)
    #   ambiguity btw cartesian indexing and linear indexing in 1d when
    #   indices may be nontraditional
    # TODO should this be implemented in ArrayInterface with vectorization?
    #@test_throws ArgumentError Base._sub2ind((1:3,), 2)
    #@test_throws ArgumentError Base._ind2sub((1:3,), 2)
    x = Array{Int,2}(undef, (2, 2))
    ArrayInterfaceCore.unsafe_setindex!(x, 1, 2, 2)
    @test ArrayInterfaceCore.unsafe_getindex(x, 2, 2) === 1

    # FIXME @test_throws MethodError ArrayInterfaceCore.unsafe_set_element!(x, 1, (:x, :x))
    # FIXME @test_throws MethodError ArrayInterfaceCore.unsafe_get_element(x, (:x, :x))
end

@testset "2-dimensional" begin
    k = 0
    cartesian = CartesianIndices((4,3))
    linear = LinearIndices(cartesian)
    for j = 1:3, i = 1:4
        k += 1
        @test @inferred(ArrayInterfaceCore.getindex(linear, i, j)) == ArrayInterfaceCore.getindex(linear, k) == k
        @test @inferred(ArrayInterfaceCore.getindex(cartesian, k)) == CartesianIndex(i,j)
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices(map(Base.Slice, (0:3,3:5))), i-1, j+2)) == k
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices(map(Base.Slice, (0:3,3:5))), k)) == CartesianIndex(i-1,j+2)
    end

    x = LinearIndices(map(Base.Slice, (static(0):static(3),static(3):static(5),static(-2):static(0))));
    @test @inferred(ArrayInterfaceCore.getindex(x, 0, 3, -2)) === 1
    @test @inferred(ArrayInterfaceCore.getindex(x, static(0), static(3), static(-2))) === 1

    @test @inferred(ArrayInterfaceCore.getindex(linear, linear)) == linear
    @test @inferred(ArrayInterfaceCore.getindex(linear, vec(linear))) == vec(linear)
    @test @inferred(ArrayInterfaceCore.getindex(linear, cartesian)) == linear
    @test @inferred(ArrayInterfaceCore.getindex(linear, vec(cartesian))) == vec(linear)
    @test @inferred(ArrayInterfaceCore.getindex(cartesian, linear)) == cartesian
    @test @inferred(ArrayInterfaceCore.getindex(cartesian, vec(linear))) == vec(cartesian)
    @test @inferred(ArrayInterfaceCore.getindex(cartesian, cartesian)) == cartesian
    @test @inferred(ArrayInterfaceCore.getindex(cartesian, vec(cartesian))) == vec(cartesian)
    @test @inferred(ArrayInterfaceCore.getindex(linear, 2:3)) === 2:3
    @test @inferred(ArrayInterfaceCore.getindex(linear, 3:-1:1)) === 3:-1:1
    @test_throws BoundsError ArrayInterfaceCore.getindex(linear, 4:13)
end

@testset "3-dimensional" begin
    l = 0
    for k = 1:2, j = 1:3, i = 1:4
        l += 1
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((4,3,2)),i,j,k)) == l
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((4,3,2)),l)) == l
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices((4,3,2)),i,j,k)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices((4,3,2)),l)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((1:4,1:3,1:2)),i,j,k)) == l
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices((1:4,1:3,1:2)),l)) == l
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices((1:4,1:3,1:2)),i,j,k)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices((1:4,1:3,1:2)),l)) == CartesianIndex(i,j,k)
    end

    l = 0
    for k = -101:-100, j = 3:5, i = 0:3
        l += 1
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices(map(Base.Slice, (0:3,3:5,-101:-100))),i,j,k)) == l
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices(map(Base.Slice, (0:3,3:5,-101:-100))),l)) == l
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices(map(Base.Slice, (0:3,3:5,-101:-100))),i,j,k)) == CartesianIndex(i,j,k)
        @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices(map(Base.Slice, (0:3,3:5,-101:-100))),l)) == CartesianIndex(i,j,k)
    end

    local A = reshape(Vector(1:9), (3,3))
    @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices(size(A)),6)) == CartesianIndex(3,2)
    @test @inferred(ArrayInterfaceCore.getindex(LinearIndices(size(A)),3, 2)) == 6
    @test @inferred(ArrayInterfaceCore.getindex(CartesianIndices(A),6)) == CartesianIndex(3,2)
    @test @inferred(ArrayInterfaceCore.getindex(LinearIndices(A),3, 2)) == 6
    for i in 1:length(A)
        @test @inferred(ArrayInterfaceCore.getindex(LinearIndices(A),ArrayInterfaceCore.getindex(CartesianIndices(A),i))) == i
    end
end

