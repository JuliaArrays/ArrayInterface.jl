
@testset "Range Interface" begin
    @testset "Range Constructors" begin
        @test @inferred(StaticInt(1):StaticInt(10)) == 1:10
        @test @inferred(StaticInt(1):StaticInt(2):StaticInt(10)) == 1:2:10
        @test @inferred(1:StaticInt(2):StaticInt(10)) == 1:2:10
        @test @inferred(StaticInt(1):StaticInt(2):10) == 1:2:10
        @test @inferred(StaticInt(1):2:StaticInt(10)) == 1:2:10
        @test @inferred(1:2:StaticInt(10)) == 1:2:10
        @test @inferred(1:StaticInt(2):10) == 1:2:10
        @test @inferred(StaticInt(1):2:10) == 1:2:10
        @test @inferred(StaticInt(1):UInt(10)) === StaticInt(1):10
        @test @inferred(UInt(1):StaticInt(1):StaticInt(10)) === 1:StaticInt(10)
        @test @inferred(ArrayInterface.OptionallyStaticUnitRange{Int,Int}(1:10)) == 1:10
        @test @inferred(ArrayInterface.OptionallyStaticUnitRange(1:10)) == 1:10

        @inferred(ArrayInterface.OptionallyStaticUnitRange(1:10))

        @test @inferred(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), static(1), static(1))) == 1:1:1
        @test @inferred(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), 1, UInt(10))) == StaticInt(1):1:10
        @test @inferred(ArrayInterface.OptionallyStaticStepRange(UInt(1), 1, StaticInt(10))) == StaticInt(1):1:10
        @test @inferred(ArrayInterface.OptionallyStaticStepRange(1:10)) == 1:1:10

        @test_throws ArgumentError ArrayInterface.OptionallyStaticUnitRange(1:2:10)
        @test_throws ArgumentError ArrayInterface.OptionallyStaticUnitRange{Int,Int}(1:2:10)
        @test_throws ArgumentError ArrayInterface.OptionallyStaticStepRange(1, 0, 10)

        @test @inferred(StaticInt(1):StaticInt(1):StaticInt(10)) === ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), StaticInt(10))
        @test @inferred(StaticInt(1):StaticInt(1):10) === ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), 10)
        @test @inferred(1:StaticInt(1):10) === ArrayInterface.OptionallyStaticUnitRange(1, 10)
        @test length(StaticInt{-1}():StaticInt{-1}():StaticInt{-10}()) == 10

        @test UnitRange(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), StaticInt(10))) === UnitRange(1, 10)
        @test UnitRange{Int}(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), StaticInt(10))) === UnitRange(1, 10)

        @test AbstractUnitRange{Int}(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), StaticInt(10))) isa ArrayInterface.OptionallyStaticUnitRange
        @test AbstractUnitRange{UInt}(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), StaticInt(10))) isa Base.OneTo
        @test AbstractUnitRange{UInt}(ArrayInterface.OptionallyStaticUnitRange(StaticInt(2), StaticInt(10))) isa UnitRange

        @test @inferred((StaticInt(1):StaticInt(10))[StaticInt(2):StaticInt(3)]) === StaticInt(2):StaticInt(3)
        @test @inferred((StaticInt(1):StaticInt(10))[StaticInt(2):3]) === StaticInt(2):3
        @test @inferred((StaticInt(1):StaticInt(10))[2:3]) === 2:3
        @test @inferred((1:StaticInt(10))[StaticInt(2):StaticInt(3)]) === 2:3

        @test -(StaticInt{1}():StaticInt{10}()) === StaticInt{-1}():StaticInt{-1}():StaticInt{-10}()

        @test reverse(StaticInt{1}():StaticInt{10}()) === StaticInt{10}():StaticInt{-1}():StaticInt{1}()
        @test reverse(StaticInt{1}():StaticInt{2}():StaticInt{9}()) === StaticInt{9}():StaticInt{-2}():StaticInt{1}()
    end

    @test isnothing(@inferred(ArrayInterface.known_first(typeof(1:4))))
    @test isone(@inferred(ArrayInterface.known_first(Base.OneTo(4))))
    @test isone(@inferred(ArrayInterface.known_first(typeof(Base.OneTo(4)))))
    @test isone(@inferred(ArrayInterface.known_first(typeof(StaticInt(1):2:10))))

    @test isnothing(@inferred(ArrayInterface.known_last(1:4)))
    @test isnothing(@inferred(ArrayInterface.known_last(typeof(1:4))))
    @test isone(@inferred(ArrayInterface.known_last(typeof(StaticInt(-1):StaticInt(2):StaticInt(1)))))

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
        @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), 10))) == 10
        @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(StaticInt(0), 10))) == 11
        @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), StaticInt(10)))) == 10
        @test @inferred(length(ArrayInterface.OptionallyStaticUnitRange(StaticInt(0), StaticInt(10)))) == 11

        @test @inferred(length(StaticInt(1):StaticInt(2):StaticInt(0))) == 0
        @test @inferred(length(StaticInt(0):StaticInt(-2):StaticInt(1))) == 0

        @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), 2, 10)))) === nothing
        @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), StaticInt(1), StaticInt(10))))) === 10
        @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(StaticInt(2), StaticInt(1), StaticInt(10))))) === 9
        @test @inferred(ArrayInterface.known_length(typeof(ArrayInterface.OptionallyStaticStepRange(StaticInt(2), StaticInt(2), StaticInt(10))))) === 5
        @test @inferred(ArrayInterface.known_length(Int)) === 1

        @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), 2, 10))) == 5
        @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), StaticInt(1), StaticInt(10)))) == 10
        @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(StaticInt(2), StaticInt(1), StaticInt(10)))) == 9
        @test @inferred(length(ArrayInterface.OptionallyStaticStepRange(StaticInt(2), StaticInt(2), StaticInt(10)))) == 5
    end

    @test @inferred(getindex(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), 10), 1)) == 1
    @test @inferred(getindex(ArrayInterface.OptionallyStaticUnitRange(StaticInt(0), 10), 1)) == 0
    @test_throws BoundsError getindex(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), 10), 0)
    @test_throws BoundsError getindex(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), 2, 10), 0)
    @test_throws BoundsError getindex(ArrayInterface.OptionallyStaticUnitRange(StaticInt(1), 10), 11)
    @test_throws BoundsError getindex(ArrayInterface.OptionallyStaticStepRange(StaticInt(1), 2, 10), 11)

    @test ArrayInterface.static_first(Base.OneTo(one(UInt))) === static(1)
    @test ArrayInterface.static_step(Base.OneTo(one(UInt))) === static(1)
    
    @test Base.setindex(1:5, [6,2], 1:2) == [6,2,3,4,5]
end

