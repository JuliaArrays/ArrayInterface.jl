
@testset "indexing" begin
    static_argdims(x) = Val(ArrayInterface.argdims(IndexLinear(), x))
    @test @inferred(static_argdims((1, CartesianIndex(1,2)))) === Val((0, 2))
    @test @inferred(static_argdims((1, [CartesianIndex(1,2), CartesianIndex(1,3)]))) === Val((0, 2))
    @test @inferred(static_argdims((1, CartesianIndex((2,2))))) === Val((0, 2))
    @test @inferred(static_argdims((CartesianIndex((2,2)), :, :))) === Val((2, 1, 1))
end

