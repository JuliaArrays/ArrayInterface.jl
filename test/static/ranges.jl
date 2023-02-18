
@test isone(@inferred(ArrayInterface.known_first(typeof(static(1):2:10))))
@test isone(@inferred(ArrayInterface.known_last(typeof(static(-1):static(2):static(1)))))

# CartesianIndices
CI = CartesianIndices((static(1):static(2), static(1):static(2)))
@test @inferred(ArrayInterface.known_last(typeof(CI))) == CartesianIndex(2, 2)

@testset "length" begin
    @test @inferred(ArrayInterface.known_length(typeof(Static.OptionallyStaticStepRange(static(1), 2, 10)))) === nothing
    @test @inferred(ArrayInterface.known_length(typeof(Static.SOneTo{-10}()))) === 0
    @test @inferred(ArrayInterface.known_length(typeof(Static.OptionallyStaticStepRange(static(1), static(1), static(10))))) === 10
    @test @inferred(ArrayInterface.known_length(typeof(Static.OptionallyStaticStepRange(static(2), static(1), static(10))))) === 9
    @test @inferred(ArrayInterface.known_length(typeof(Static.OptionallyStaticStepRange(static(2), static(2), static(10))))) === 5
    @test @inferred(ArrayInterface.known_length(Int)) === 1
end

@testset "indices" begin
    A23 = ones(2,3);
    SA23 = MArray(A23);
    A32 = ones(3,2);
    SA32 = MArray(A32);

    @test @inferred(ArrayInterface.indices(A23, (static(1),static(2)))) === (Base.Slice(StaticInt(1):2), Base.Slice(StaticInt(1):3))
    @test @inferred(ArrayInterface.indices((A23, A32))) == 1:6
    @test @inferred(ArrayInterface.indices((SA23, A32))) == 1:6
    @test @inferred(ArrayInterface.indices((A23, SA32))) == 1:6
    @test @inferred(ArrayInterface.indices((SA23, SA32))) == 1:6
    @test @inferred(ArrayInterface.indices(A23)) == 1:6
    @test @inferred(ArrayInterface.indices(SA23)) == 1:6
    @test @inferred(ArrayInterface.indices(A23, 1)) == 1:2
    @test @inferred(ArrayInterface.indices(SA23, StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, A32), (1, 2))) == 1:2
    @test @inferred(ArrayInterface.indices((SA23, A32), (StaticInt(1), 2))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, SA32), (1, StaticInt(2)))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((SA23, SA32), (StaticInt(1), StaticInt(2)))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, A23), 1)) == 1:2
    @test @inferred(ArrayInterface.indices((SA23, SA23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((SA23, A23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, SA23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((SA23, SA23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))

    @test_throws ErrorException ArrayInterface.indices((A23, ones(3, 3)), 1)
    @test_throws ErrorException ArrayInterface.indices((A23, ones(3, 3)), (1, 2))
    @test_throws ErrorException ArrayInterface.indices((SA23, ones(3, 3)), StaticInt(1))
    @test_throws ErrorException ArrayInterface.indices((SA23, ones(3, 3)), (StaticInt(1), 2))
    @test_throws ErrorException ArrayInterface.indices((SA23, SA23), (StaticInt(1), StaticInt(2)))

    @test size(similar(ones(2, 4), ArrayInterface.indices(ones(2, 4), 1), ArrayInterface.indices(ones(2, 4), 2))) == (2, 4)
    @test axes(ArrayInterface.indices(ones(2,2))) === (StaticInt(1):4,)
    @test axes(Base.Slice(StaticInt(2):4)) === (Base.IdentityUnitRange(StaticInt(2):4),)
    @test Base.axes1(ArrayInterface.indices(ones(2,2))) === StaticInt(1):4
    @test Base.axes1(Base.Slice(StaticInt(2):4)) === Base.IdentityUnitRange(StaticInt(2):4)

    x = vec(A23); y = vec(A32);
    @test ArrayInterface.indices((x',y'),StaticInt(1)) === Base.Slice(StaticInt(1):StaticInt(1))
    @test ArrayInterface.indices((x,y), StaticInt(2)) === Base.Slice(StaticInt(1):StaticInt(1))
end
