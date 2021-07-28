@testset "restructure" begin
    x = rand(Float32, 2, 2)
    y = rand(4)
    yr = ArrayInterface.restructure(x, y)
    @test yr isa Matrix{Float64}
    @test size(yr) == (2,2)
    @test vec(yr) == vec(y)

    @testset "views" begin
        x = @view rand(4)[1:2]
        y = rand(2)
        yr = ArrayInterface.restructure(x, y)
        @test yr isa Vector{Float64}
        @test size(yr) == (2,)
        @test yr == y

        x = @view rand(4,4)[1:2,1:2]
        y = rand(2,2)
        yr = ArrayInterface.restructure(x, y)
        @test yr isa Matrix{Float64}
        @test size(yr) == (2,2)
        @test yr == y


        x = @view rand(4,4)[1]
        y = @view rand(2,2)[1]
        yr = ArrayInterface.restructure(x, y)
        @test yr isa Array{Float64,0}
        @test size(yr) == ()
        @test yr == y
    end
end