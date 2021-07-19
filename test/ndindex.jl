

@testset "NDIndex" begin

x = 
y = NDIndex((1,static(2),3))
z = NDIndex(static(3), static(3), static(3))

@testset "constructors" begin
    @test static(CartesianIndex(3, 3, 3)) === z == Base.setindex(Base.setindex(x, 3, 1), 3, 2)
    @test @inferred(ArrayInterface.Static.dynamic(z)) === CartesianIndex(3, 3, 3)
    @test @inferred(ArrayInterface.Static.known(z)) === (3, 3, 3)
    @test Tuple(@inferred(NDIndex{0}())) === ()
    @test @inferred(NDIndex{3}(1, static(2), 3)) === y
    @test @inferred(NDIndex{3}((1, static(2), 3))) === y
    @test @inferred(NDIndex{3}((1, static(2)))) === NDIndex(1, static(2), static(1))
    @test @inferred(NDIndex(x, y)) === NDIndex(1, 2, 3, 1, static(2), 3)
    @test @inferred(NDIndex{3,Tuple{Int,Int,Int}}((1,2, 3))) === x
end

@test @inferred(length(x)) === 3
@test @inferred(length(typeof(x))) === 3
@test @inferred(y[2]) === 2
@test @inferred(y[static(2)]) === static(2)

@test @inferred(-y) === NDIndex((-1,-static(2),-3))
@test @inferred(y + y) === NDIndex((2,static(4),6))
@test @inferred(y - y) === NDIndex((0,static(0),0))
@test @inferred(zero(x)) === NDIndex(static(0),static(0),static(0))
@test @inferred(oneunit(x)) === NDIndex(static(1),static(1),static(1))
@test @inferred(x * 3) === NDIndex((3,6,9))
@test @inferred(3 * x) === NDIndex((3,6,9))

@test @inferred(min(x, z)) === x
@test @inferred(max(x, z)) === NDIndex(3, 3, 3)
@test !@inferred(isless(y, x))
@test @inferred(isless(x, z))
@test @inferred(ArrayInterface.Static.lt(oneunit(z), z)) === static(true)


end

