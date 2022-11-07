
@testset "Range Constructors" begin
    @test @inferred(static(1):static(10)) == 1:10
    @test @inferred(ArrayInterface.SUnitRange{1,10}()) == 1:10
    @test @inferred(static(1):static(2):static(10)) == 1:2:10
    @test @inferred(1:static(2):static(10)) == 1:2:10
    @test @inferred(static(1):static(2):10) == 1:2:10
    @test @inferred(static(1):2:static(10)) == 1:2:10
    @test @inferred(1:2:static(10)) == 1:2:10
    @test @inferred(1:static(2):10) == 1:2:10
    @test @inferred(static(1):2:10) == 1:2:10
    @test @inferred(static(1):UInt(10)) === static(1):10
    @test @inferred(UInt(1):static(1):static(10)) === 1:static(10)
    @test ArrayInterface.SUnitRange(1, 10) == 1:10
    @test @inferred(ArrayInterface.OptionallyStaticUnitRange{Int,Int}(1:10)) == 1:10
    @test @inferred(ArrayInterface.OptionallyStaticUnitRange(1:10)) == 1:10

    @inferred(ArrayInterface.OptionallyStaticUnitRange(1:10))

    @test @inferred(ArrayInterface.OptionallyStaticStepRange(static(1), static(1), static(1))) == 1:1:1
    @test @inferred(ArrayInterface.OptionallyStaticStepRange(static(1), 1, UInt(10))) == static(1):1:10
    @test @inferred(ArrayInterface.OptionallyStaticStepRange(UInt(1), 1, static(10))) == static(1):1:10
    @test @inferred(ArrayInterface.OptionallyStaticStepRange(1:10)) == 1:1:10

    @test_throws ArgumentError ArrayInterface.OptionallyStaticUnitRange(1:2:10)
    @test_throws ArgumentError ArrayInterface.OptionallyStaticUnitRange{Int,Int}(1:2:10)
    @test_throws ArgumentError ArrayInterface.OptionallyStaticStepRange(1, 0, 10)

    @test @inferred(static(1):static(1):static(10)) === ArrayInterface.OptionallyStaticUnitRange(static(1), static(10))
    @test @inferred(static(1):static(1):10) === ArrayInterface.OptionallyStaticUnitRange(static(1), 10)
    @test @inferred(1:static(1):10) === ArrayInterface.OptionallyStaticUnitRange(1, 10)
    @test length(static(-1):static(-1):static(-10)) == 10 == lastindex(static(-1):static(-1):static(-10))

    @test UnitRange(ArrayInterface.OptionallyStaticUnitRange(static(1), static(10))) === UnitRange(1, 10)
    @test UnitRange{Int}(ArrayInterface.OptionallyStaticUnitRange(static(1), static(10))) === UnitRange(1, 10)

    @test AbstractUnitRange{Int}(ArrayInterface.OptionallyStaticUnitRange(static(1), static(10))) isa ArrayInterface.OptionallyStaticUnitRange
    @test AbstractUnitRange{UInt}(ArrayInterface.OptionallyStaticUnitRange(static(1), static(10))) isa Base.OneTo
    @test AbstractUnitRange{UInt}(ArrayInterface.OptionallyStaticUnitRange(static(2), static(10))) isa UnitRange

    @test @inferred((static(1):static(10))[static(2):static(3)]) === static(2):static(3)
    @test @inferred((static(1):static(10))[static(2):3]) === static(2):3
    @test @inferred((static(1):static(10))[2:3]) === 2:3
    @test @inferred((1:static(10))[static(2):static(3)]) === 2:3

    @test Base.checkindex(Bool, static(1):static(10), static(1):static(5))
    @test -(static(1):static(10)) === static(-1):static(-1):static(-10)

    @test reverse(static(1):static(10)) === static(10):static(-1):static(1)
    @test reverse(static(1):static(2):static(9)) === static(9):static(-2):static(1)
end

# iteration
@test iterate(static(1):static(5), 5) === nothing
@test iterate(static(2):static(5), 5) === nothing

@test isone(@inferred(ArrayInterface.known_first(typeof(static(1):2:10))))
@test isone(@inferred(ArrayInterface.known_last(typeof(static(-1):static(2):static(1)))))

# CartesianIndices
CI = CartesianIndices((static(1):static(2), static(1):static(2)))
@test @inferred(ArrayInterface.known_last(typeof(CI))) == CartesianIndex(2, 2)

@testset "length" begin
    @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(1, 0))) == 0
    @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(1, 10))) == 10
    @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(static(1), 10))) == 10
    @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(static(0), 10))) == 11
    @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(static(1), static(10)))) == 10
    @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(static(0), static(10)))) == 11

    @test @inferred(length(static(1):static(2):static(0))) == 0
    @test @inferred(length(static(0):static(-2):static(1))) == 0

    @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(static(1), 2, 10)))) === nothing
    @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.SOneTo{-10}()))) === 0
    @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(static(1), static(1), static(10))))) === 10
    @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(static(2), static(1), static(10))))) === 9
    @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(static(2), static(2), static(10))))) === 5
    @test @inferred(ArrayInterface.known_length(Int)) === 1

    @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(static(1), 2, 10))) == 5
    @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(static(1), static(1), static(10)))) == 10
    @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(static(2), static(1), static(10)))) == 9
    @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(static(2), static(2), static(10)))) == 5
end

@test @inferred(getindex(ArrayInterface.OptionallyStaticUnitRange(static(1), 10), 1)) == 1
@test @inferred(getindex(ArrayInterface.OptionallyStaticUnitRange(static(0), 10), 1)) == 0
@test_throws BoundsError getindex(ArrayInterface.OptionallyStaticUnitRange(static(1), 10), 0)
@test_throws BoundsError getindex(ArrayInterface.OptionallyStaticStepRange(static(1), 2, 10), 0)
@test_throws BoundsError getindex(ArrayInterface.OptionallyStaticUnitRange(static(1), 10), 11)
@test_throws BoundsError getindex(ArrayInterface.OptionallyStaticStepRange(static(1), 2, 10), 11)

@test ArrayInterface.static_first(Base.OneTo(one(UInt))) === static(1)
@test ArrayInterface.static_step(Base.OneTo(one(UInt))) === static(1)

@test @inferred(eachindex(static(-7):static(7))) === static(1):static(15)
@test @inferred((static(-7):static(7))[first(eachindex(static(-7):static(7)))]) == -7

@test @inferred(firstindex(128:static(-1):1)) == 1

@test identity.(static(1):5) isa Vector{Int}
@test (static(1):5) .+ (1:3)' isa Matrix{Int}
@test similar(Array{Int}, (static(1):(4),)) isa Vector{Int}
@test similar(Array{Int}, (static(1):(4), Base.OneTo(4))) isa Matrix{Int}
@test similar(Array{Int}, (Base.OneTo(4), static(1):(4))) isa Matrix{Int}

@testset "indices" begin
    A23 = ones(2,3);
    SA23 = MArray(A23);
    A32 = ones(3,2);
    SA32 = MArray(A32)

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

    @test_throws AssertionError ArrayInterface.indices((A23, ones(3, 3)), 1)
    @test_throws AssertionError ArrayInterface.indices((A23, ones(3, 3)), (1, 2))
    @test_throws AssertionError ArrayInterface.indices((SA23, ones(3, 3)), StaticInt(1))
    @test_throws AssertionError ArrayInterface.indices((SA23, ones(3, 3)), (StaticInt(1), 2))
    @test_throws AssertionError ArrayInterface.indices((SA23, SA23), (StaticInt(1), StaticInt(2)))

    @test size(similar(ones(2, 4), ArrayInterface.indices(ones(2, 4), 1), ArrayInterface.indices(ones(2, 4), 2))) == (2, 4)
    @test axes(ArrayInterface.indices(ones(2,2))) === (StaticInt(1):4,)
    @test axes(Base.Slice(StaticInt(2):4)) === (Base.IdentityUnitRange(StaticInt(2):4),)
    @test Base.axes1(ArrayInterface.indices(ones(2,2))) === StaticInt(1):4
    @test Base.axes1(Base.Slice(StaticInt(2):4)) === Base.IdentityUnitRange(StaticInt(2):4)

    x = vec(A23); y = vec(A32);
    @test ArrayInterface.indices((x',y'),StaticInt(1)) === Base.Slice(StaticInt(1):StaticInt(1))
    @test ArrayInterface.indices((x,y), StaticInt(2)) === Base.Slice(StaticInt(1):StaticInt(1))
end

