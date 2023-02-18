
@testset "to_index" begin
    axis = 1:3
    @test @inferred(ArrayInterface.to_index(axis, 1)) === 1
    @test @inferred(ArrayInterface.to_index(axis, static(1))) === static(1)
    @test @inferred(ArrayInterface.to_index(axis, CartesianIndex(1))) === (1,)
    @test @inferred(ArrayInterface.to_index(axis, 1:2)) === 1:2
    @test @inferred(ArrayInterface.to_index(axis, CartesianIndices((1:2,)))) == (1:2,)
    @test @inferred(ArrayInterface.to_index(axis, CartesianIndices((2:3,)))) == (2:3,)
    @test @inferred(ArrayInterface.to_index(axis, [1, 2])) == [1, 2]
    @test @inferred(ArrayInterface.to_index(axis, [true, false, false])) == [1]
    index = @inferred(ArrayInterface.to_index(axis, :))
    @test @inferred(ArrayInterface.to_index(axis, index)) == index == ArrayInterface.indices(axis)

    x = LinearIndices((static(0):static(3), static(3):static(5), static(-2):static(0)))
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
    a = ones(2, 2, 1)
    v = ones(2)

    @testset "linear indexing" begin
        @test @inferred(ArrayInterface.static_to_indices(a, (1,))) == (1,)
        @test @inferred(ArrayInterface.static_to_indices(a, (1:2,))) == (1:2,)

        @testset "Linear indexing doesn't ruin vector indexing" begin
            @test @inferred(ArrayInterface.static_to_indices(v, (1:2,))) == (1:2,)
            @test @inferred(ArrayInterface.static_to_indices(v, (1,))) == (1,)
        end
    end

    #@test @inferred(ArrayInterface.static_to_indices(a, (CartesianIndices(()),))) == (CartesianIndices(()),)
    inds = @inferred(ArrayInterface.static_to_indices(a, (:,)))
    @test @inferred(ArrayInterface.static_to_indices(a, inds)) == inds == (ArrayInterface.indices(a),)
    @test @inferred(ArrayInterface.static_to_indices(ones(2, 2, 2), ([true, true], CartesianIndex(1, 1)))) == ([1, 2], 1, 1)
    @test @inferred(ArrayInterface.static_to_indices(a, (1, 1))) == (1, 1)
    @test @inferred(ArrayInterface.static_to_indices(a, (1, CartesianIndex(1)))) == (1, 1)
    @test @inferred(ArrayInterface.static_to_indices(a, (1, false))) == (1, 0)
    @test @inferred(ArrayInterface.static_to_indices(a, (1, 1:2))) == (1, 1:2)
    @test @inferred(ArrayInterface.static_to_indices(a, (1:2, 1))) == (1:2, 1)
    @test @inferred(ArrayInterface.static_to_indices(a, (1, :))) == (1, Base.Slice(1:2))
    @test @inferred(ArrayInterface.static_to_indices(a, (:, 1))) == (Base.Slice(1:2), 1)
    @test @inferred(ArrayInterface.static_to_indices(a, ([true, true], :))) == (Base.LogicalIndex(Bool[1, 1]), Base.Slice(1:2))
    @test @inferred(ArrayInterface.static_to_indices(a, (CartesianIndices((1,)), 1))) == (1:1, 1)
    @test @inferred(ArrayInterface.static_to_indices(a, (1, 1, 1))) == (1, 1, 1)
    @test @inferred(ArrayInterface.static_to_indices(a, (1, fill(true, 2, 2)))) == Base.to_indices(a, (1, fill(true, 2, 2)))
    inds = @inferred(ArrayInterface.static_to_indices(a, (fill(true, 2, 2, 1),)))
    # Conversion to LogicalIndex doesn't change
    @test @inferred(ArrayInterface.static_to_indices(a, inds)) == inds
    @test @inferred(ArrayInterface.static_to_indices(a, (fill(true, 2, 1), 1))) == Base.to_indices(a, (fill(true, 2, 1), 1))
    @test @inferred(ArrayInterface.static_to_indices(a, (fill(true, 2), 1, 1))) == Base.to_indices(a, (fill(true, 2), 1, 1))
    @test @inferred(ArrayInterface.static_to_indices(a, ([CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1)],))) == (CartesianIndex{3}[CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1)],)
    @test @inferred(ArrayInterface.static_to_indices(a, ([CartesianIndex(1, 1), CartesianIndex(1, 2)], 1:1))) == (CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(1, 2)], 1:1)
    @test @inferred(first(ArrayInterface.static_to_indices(a, (fill(true, 2, 2, 1),)))) isa Base.LogicalIndex
end

@testset "splat indexing" begin
    struct SplatFirst end

    ArrayInterface.to_index(x, ::SplatFirst) = map(first, axes(x))
    ArrayInterface.is_splat_index(::Type{SplatFirst}) = true
    x = rand(4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
    i = (1, SplatFirst(), 2, SplatFirst(), CartesianIndex(1, 1))

    @test @inferred(ArrayInterface.static_to_indices(x, i)) == (1, 1, 1, 1, 1, 1, 2, 1, 1, 1)
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

    @test @inferred(ArrayInterface.to_axes(A, (), (inds,))) === (inds,)
end

@testset  "getindex with additional inds" begin
    A = reshape(1:12, (3, 4))
    subA = view(A, :, :)
    LA = LinearIndices(A)
    CA = CartesianIndices(A)
    @test @inferred(ArrayInterface.static_getindex(A, 1, 1, 1)) == 1
    @test @inferred(ArrayInterface.static_getindex(A, 1, 1, :)) == [1]
    @test @inferred(ArrayInterface.static_getindex(A, 1, 1, 1:1)) == [1]
    @test @inferred(ArrayInterface.static_getindex(A, 1, 1, :, :)) == ones(1, 1)
    @test @inferred(ArrayInterface.static_getindex(A, :, 1, 1)) == 1:3
    @test @inferred(ArrayInterface.static_getindex(A, 2:3, 1, 1)) == 2:3
    @test @inferred(ArrayInterface.static_getindex(A, static(1):2, 1, 1)) == 1:2
    @test @inferred(ArrayInterface.static_getindex(A, :, 1, :)) == reshape(1:3, 3, 1)
    @test @inferred(ArrayInterface.static_getindex(subA, 1, 1, 1)) == 1
    @test @inferred(ArrayInterface.static_getindex(subA, 1, 1, :)) == [1]
    @test @inferred(ArrayInterface.static_getindex(subA, 1, 1, 1:1)) == [1]
    @test @inferred(ArrayInterface.static_getindex(subA, 1, 1, :, :)) == ones(1, 1)
    @test @inferred(ArrayInterface.static_getindex(subA, :, 1, 1)) == 1:3
    @test @inferred(ArrayInterface.static_getindex(subA, 2:3, 1, 1)) == 2:3
    @test @inferred(ArrayInterface.static_getindex(subA, static(1):2, 1, 1)) == 1:2
    @test @inferred(ArrayInterface.static_getindex(subA, :, 1, :)) == reshape(1:3, 3, 1)
    @test @inferred(ArrayInterface.static_getindex(LA, 1, 1, 1)) == 1
    @test @inferred(ArrayInterface.static_getindex(LA, 1, 1, :)) == [1]
    @test @inferred(ArrayInterface.static_getindex(LA, 1, 1, 1:1)) == [1]
    @test @inferred(ArrayInterface.static_getindex(LA, 1, 1, :, :)) == ones(1, 1)
    @test @inferred(ArrayInterface.static_getindex(LA, :, 1, 1)) == 1:3
    @test @inferred(ArrayInterface.static_getindex(LA, 2:3, 1, 1)) == 2:3
    @test @inferred(ArrayInterface.static_getindex(LA, static(1):2, 1, 1)) == 1:2
    @test @inferred(ArrayInterface.static_getindex(LA, :, 1, :)) == reshape(1:3, 3, 1)
    @test @inferred(ArrayInterface.static_getindex(CA, 1, 1, 1)) == CartesianIndex(1, 1)
    @test @inferred(ArrayInterface.static_getindex(CA, 1, 1, :)) == [CartesianIndex(1, 1)]
    @test @inferred(ArrayInterface.static_getindex(CA, 1, 1, 1:1)) == [CartesianIndex(1, 1)]
    @test @inferred(ArrayInterface.static_getindex(CA, 1, 1, :, :)) == fill(CartesianIndex(1, 1), 1, 1)
    @test @inferred(ArrayInterface.static_getindex(CA, :, 1, 1)) ==
        reshape(CartesianIndex(1, 1):CartesianIndex(3, 1), 3)
    @test @inferred(ArrayInterface.static_getindex(CA, 2:3, 1, 1)) ==
        reshape(CartesianIndex(2, 1):CartesianIndex(3, 1), 2)
    @test @inferred(ArrayInterface.static_getindex(CA, static(1):2, 1, 1)) ==
        reshape(CartesianIndex(1, 1):CartesianIndex(2, 1), 2)
    @test @inferred(ArrayInterface.static_getindex(CA, :, 1, :)) ==
        reshape(CartesianIndex(1, 1):CartesianIndex(3, 1), 3, 1)
end

@testset "0-dimensional" begin
    x = Array{Int,0}(undef)
    ArrayInterface.setindex!(x, 1)
    @test @inferred(ArrayInterface.static_getindex(x)) == 1
end

@testset "1-dimensional" begin
    for i = 1:3
        @test @inferred(ArrayInterface.static_getindex(LinearIndices((3,)), i)) == i
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices((3,)), i)) == CartesianIndex(i,)
    end
    @test @inferred(ArrayInterface.static_getindex(LinearIndices((3,)), 2, 1)) == 2
    @test @inferred(ArrayInterface.static_getindex(LinearIndices((3,)), [1])) == [1]
    # !!NOTE!! this is different than Base.getindex(::LinearIndices, ::AbstractUnitRange)
    # which returns a UnitRange. Instead we try to preserve axes if at all possible so the
    # values are the same but it's still wrapped in LinearIndices struct
    @test @inferred(ArrayInterface.static_getindex(LinearIndices((3,)), 1:2)) == 1:2
    @test @inferred(ArrayInterface.static_getindex(LinearIndices((3,)), 1:2:3)) === 1:2:3
    @test_throws BoundsError ArrayInterface.static_getindex(LinearIndices((3,)), 2:4)
    @test_throws BoundsError ArrayInterface.static_getindex(CartesianIndices((3,)), 2, 2)
    #   ambiguity btw cartesian indexing and linear indexing in 1d when
    #   indices may be nontraditional
    # TODO should this be implemented in ArrayInterface with vectorization?
    #@test_throws ArgumentError Base._sub2ind((1:3,), 2)
    #@test_throws ArgumentError Base._ind2sub((1:3,), 2)
    x = Array{Int,2}(undef, (2, 2))
    ArrayInterface.unsafe_setindex!(x, 1, 2, 2)
    @test ArrayInterface.unsafe_getindex(x, 2, 2) === 1

    # FIXME @test_throws MethodError ArrayInterface.unsafe_set_element!(x, 1, (:x, :x))
    # FIXME @test_throws MethodError ArrayInterface.unsafe_get_element(x, (:x, :x))
end

@testset "2-dimensional" begin
    k = 0
    cartesian = CartesianIndices((4, 3))
    linear = LinearIndices(cartesian)
    for j = 1:3, i = 1:4
        k += 1
        @test @inferred(ArrayInterface.static_getindex(linear, i, j)) == ArrayInterface.static_getindex(linear, k) == k
        @test @inferred(ArrayInterface.static_getindex(cartesian, k)) == CartesianIndex(i, j)
        @test @inferred(ArrayInterface.static_getindex(LinearIndices(map(Base.Slice, (0:3, 3:5))), i - 1, j + 2)) == k
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices(map(Base.Slice, (0:3, 3:5))), k)) == CartesianIndex(i - 1, j + 2)
    end

    x = LinearIndices(map(Base.Slice, (static(0):static(3), static(3):static(5), static(-2):static(0))))
    
    if VERSION >= v"1.7"
        @test @inferred(ArrayInterface.static_getindex(x, 0, 3, -2)) === 1
    else
        @test ArrayInterface.static_getindex(x, 0, 3, -2) === 1
    end
    
    @test @inferred(ArrayInterface.static_getindex(x, static(0), static(3), static(-2))) === 1

    @test @inferred(ArrayInterface.static_getindex(linear, linear)) == linear
    @test @inferred(ArrayInterface.static_getindex(linear, vec(linear))) == vec(linear)
    @test @inferred(ArrayInterface.static_getindex(linear, cartesian)) == linear
    @test @inferred(ArrayInterface.static_getindex(linear, vec(cartesian))) == vec(linear)
    @test @inferred(ArrayInterface.static_getindex(cartesian, linear)) == cartesian
    @test @inferred(ArrayInterface.static_getindex(cartesian, vec(linear))) == vec(cartesian)
    @test @inferred(ArrayInterface.static_getindex(cartesian, cartesian)) == cartesian
    @test @inferred(ArrayInterface.static_getindex(cartesian, vec(cartesian))) == vec(cartesian)
    @test @inferred(ArrayInterface.static_getindex(linear, 2:3)) === 2:3
    @test @inferred(ArrayInterface.static_getindex(linear, 3:-1:1)) === 3:-1:1
    @test @inferred(ArrayInterface.static_getindex(linear, >(1), <(3))) == linear[(begin+1):end, 1:(end-1)]
    @test @inferred(ArrayInterface.static_getindex(linear, >=(1), <=(3))) == linear[begin:end, 1:end]
    @test_throws BoundsError ArrayInterface.static_getindex(linear, 4:13)
end

@testset "3-dimensional" begin
    l = 0
    for k = 1:2, j = 1:3, i = 1:4
        l += 1
        @test @inferred(ArrayInterface.static_getindex(LinearIndices((4, 3, 2)), i, j, k)) == l
        @test @inferred(ArrayInterface.static_getindex(LinearIndices((4, 3, 2)), l)) == l
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices((4, 3, 2)), i, j, k)) == CartesianIndex(i, j, k)
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices((4, 3, 2)), l)) == CartesianIndex(i, j, k)
        @test @inferred(ArrayInterface.static_getindex(LinearIndices((1:4, 1:3, 1:2)), i, j, k)) == l
        @test @inferred(ArrayInterface.static_getindex(LinearIndices((1:4, 1:3, 1:2)), l)) == l
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices((1:4, 1:3, 1:2)), i, j, k)) == CartesianIndex(i, j, k)
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices((1:4, 1:3, 1:2)), l)) == CartesianIndex(i, j, k)
    end

    l = 0
    for k = -101:-100, j = 3:5, i = 0:3
        l += 1
        @test @inferred(ArrayInterface.static_getindex(LinearIndices(map(Base.Slice, (0:3, 3:5, -101:-100))), i, j, k)) == l
        @test @inferred(ArrayInterface.static_getindex(LinearIndices(map(Base.Slice, (0:3, 3:5, -101:-100))), l)) == l
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices(map(Base.Slice, (0:3, 3:5, -101:-100))), i, j, k)) == CartesianIndex(i, j, k)
        @test @inferred(ArrayInterface.static_getindex(CartesianIndices(map(Base.Slice, (0:3, 3:5, -101:-100))), l)) == CartesianIndex(i, j, k)
    end

    local A = reshape(Vector(1:9), (3, 3))
    @test @inferred(ArrayInterface.static_getindex(CartesianIndices(size(A)), 6)) == CartesianIndex(3, 2)
    @test @inferred(ArrayInterface.static_getindex(LinearIndices(size(A)), 3, 2)) == 6
    @test @inferred(ArrayInterface.static_getindex(CartesianIndices(A), 6)) == CartesianIndex(3, 2)
    @test @inferred(ArrayInterface.static_getindex(LinearIndices(A), 3, 2)) == 6
    for i in 1:length(A)
        @test @inferred(ArrayInterface.static_getindex(LinearIndices(A), ArrayInterface.static_getindex(CartesianIndices(A), i))) == i
    end
end

A = zeros(3, 4, 5);
A[:] = 1:60
Ap = @view(PermutedDimsArray(A, (3, 1, 2))[:, 1:2, 1])';
S = MArray(zeros(2, 3, 4))
A_trailingdim = zeros(2, 3, 4, 1)
Sp = @view(PermutedDimsArray(S, (3, 1, 2))[2:3, 1:2, :]);

Sp2 = @view(PermutedDimsArray(S, (3, 2, 1))[2:3, :, :]);

Mp = @view(PermutedDimsArray(S, (3, 1, 2))[:, 2, :])';
Mp2 = @view(PermutedDimsArray(S, (3, 1, 2))[2:3, :, 2])';

D = @view(A[:, 2:2:4, :]);
R = StaticInt(1):StaticInt(2);
Rnr = reinterpret(Int32, R);
Ar = reinterpret(Float32, A);
A2 = zeros(4, 3, 5)
A2r = reinterpret(ComplexF64, A2)

irev = Iterators.reverse(S)
igen = Iterators.map(identity, S)
iacc = Iterators.accumulate(+, S)
iprod = Iterators.product(axes(S)...)
iflat = Iterators.flatten(iprod)
ienum = enumerate(S)
ipairs = pairs(S)
izip = zip(S, S)

sv5 = MArray(zeros(5));
v5 = Vector{Float64}(undef, 5);
