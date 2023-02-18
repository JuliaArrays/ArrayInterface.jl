@testset "insert/deleteat" begin
    @test @inferred(ArrayInterface.insert([1,2,3], 2, -2)) == [1, -2, 2, 3]
    @test @inferred(ArrayInterface.deleteat([1, 2, 3], 2)) == [1, 3]

    @test @inferred(ArrayInterface.deleteat([1, 2, 3], [1, 2])) == [3]
    @test @inferred(ArrayInterface.deleteat([1, 2, 3], [1, 3])) == [2]
    @test @inferred(ArrayInterface.deleteat([1, 2, 3], [2, 3])) == [1]

    @test @inferred(ArrayInterface.insert((2,3,4), 1, -2)) == (-2, 2, 3, 4)
    @test @inferred(ArrayInterface.insert((2,3,4), 2, -2)) == (2, -2, 3, 4)
    @test @inferred(ArrayInterface.insert((2,3,4), 3, -2)) == (2, 3, -2, 4)

    @test @inferred(ArrayInterface.deleteat((2, 3, 4), 1)) == (3, 4)
    @test @inferred(ArrayInterface.deleteat((2, 3, 4), 2)) == (2, 4)
    @test @inferred(ArrayInterface.deleteat((2, 3, 4), 3)) == (2, 3)
    @test ArrayInterface.deleteat((1, 2, 3), [1, 2]) == (3,)
end