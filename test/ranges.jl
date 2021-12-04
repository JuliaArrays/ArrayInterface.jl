
@testset "Range Interface" begin
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

    @test isnothing(@inferred(ArrayInterface.known_first(typeof(1:4))))
    @test isone(@inferred(ArrayInterface.known_first(Base.OneTo(4))))
    @test isone(@inferred(ArrayInterface.known_first(typeof(Base.OneTo(4)))))
    @test isone(@inferred(ArrayInterface.known_first(typeof(static(1):2:10))))

    @test isnothing(@inferred(ArrayInterface.known_last(1:4)))
    @test isnothing(@inferred(ArrayInterface.known_last(typeof(1:4))))
    @test isone(@inferred(ArrayInterface.known_last(typeof(static(-1):static(2):static(1)))))

    # CartesianIndices
    CI = CartesianIndices((2, 2))
    @test @inferred(ArrayInterface.known_first(typeof(CI))) == CartesianIndex(1, 1)
    @test @inferred(ArrayInterface.known_last(typeof(CI))) == nothing

    CI = CartesianIndices((static(1):static(2), static(1):static(2)))
    @test @inferred(ArrayInterface.known_first(typeof(CI))) == CartesianIndex(1, 1)
    @test @inferred(ArrayInterface.known_last(typeof(CI))) == CartesianIndex(2, 2)

    @test isnothing(@inferred(ArrayInterface.known_step(typeof(1:0.2:4))))
    @test isone(@inferred(ArrayInterface.known_step(1:4)))
    @test isone(@inferred(ArrayInterface.known_step(typeof(1:4))))
    @test isone(@inferred(ArrayInterface.known_step(typeof(Base.Slice(1:4)))))
    @test isone(@inferred(ArrayInterface.known_step(typeof(view(1:4, 1:2)))))

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

    @test Base.setindex(1:5, [6,2], 1:2) == [6,2,3,4,5]

    @test @inferred(eachindex(static(-7):static(7))) === static(1):static(15)
    @test @inferred((static(-7):static(7))[first(eachindex(static(-7):static(7)))]) == -7
end

