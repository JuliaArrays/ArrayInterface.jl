using ArrayInterface, Test
using Base: setindex
import ArrayInterface: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance, is_cpu_column_major, stridelayout
@test ArrayInterface.ismutable(rand(3))

using StaticArrays
@test ArrayInterface.ismutable(@SVector [1,2,3]) == false
@test ArrayInterface.ismutable(@MVector [1,2,3]) == true
@test ArrayInterface.ismutable(1:10) == false
@test ArrayInterface.ismutable((0.1,1.0)) == false
@test isone(ArrayInterface.known_first(typeof(StaticArrays.SOneTo(7))))
@test ArrayInterface.known_last(typeof(StaticArrays.SOneTo(7))) == 7

using LinearAlgebra, SparseArrays

D=Diagonal([1,2,3,4])
@test has_sparsestruct(D)
rowind,colind=findstructralnz(D)
@test [D[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4]
@test length(rowind)==4
@test length(rowind)==length(colind)

Bu = Bidiagonal([1,2,3,4], [7,8,9], :U)
@test has_sparsestruct(Bu)
rowind,colind=findstructralnz(Bu)
@test [Bu[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,7,2,8,3,9,4]
Bl = Bidiagonal([1,2,3,4], [7,8,9], :L)
@test has_sparsestruct(Bl)
rowind,colind=findstructralnz(Bl)
@test [Bl[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,7,2,8,3,9,4]

Tri=Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])
@test has_sparsestruct(Tri)
rowind,colind=findstructralnz(Tri)
@test [Tri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4,4,5,6,1,2,3]
STri=SymTridiagonal([1,2,3,4],[5,6,7])
@test has_sparsestruct(STri)
rowind,colind=findstructralnz(STri)
@test [STri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4,5,6,7,5,6,7]

Sp=sparse([1,2,3],[1,2,3],[1,2,3])
@test has_sparsestruct(Sp)
rowind,colind=findstructralnz(Sp)
@test [Tri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3]

@test ArrayInterface.ismutable(spzeros(1, 1))
@test ArrayInterface.ismutable(spzeros(1))

@test !fast_scalar_indexing(qr(rand(10, 10)).Q)
@test !fast_scalar_indexing(qr(rand(10, 10), Val(true)).Q)
@test !fast_scalar_indexing(lq(rand(10, 10)).Q)

using BandedMatrices

B=BandedMatrix(Ones(5,5), (-1,2))
B[band(1)].=[1,2,3,4]
B[band(2)].=[5,6,7]
@test has_sparsestruct(B)
rowind,colind=findstructralnz(B)
@test [B[rowind[i],colind[i]] for i in 1:length(rowind)]==[5,6,7,1,2,3,4]
B=BandedMatrix(Ones(4,6), (-1,2))
B[band(1)].=[1,2,3,4]
B[band(2)].=[5,6,7,8]
rowind,colind=findstructralnz(B)
@test [B[rowind[i],colind[i]] for i in 1:length(rowind)]==[5,6,7,8,1,2,3,4]

using BlockBandedMatrices
BB=BlockBandedMatrix(Ones(10,10),[1,2,3,4],[4,3,2,1],(1,0))
BB[Block(1,1)].=[1 2 3 4]
BB[Block(2,1)].=[5 6 7 8;9 10 11 12]
rowind,colind=findstructralnz(BB)
@test [BB[rowind[i],colind[i]] for i in 1:length(rowind)]==
    [1,5,9,2,6,10,3,7,11,4,8,12,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1]

dense=collect(Ones(8,8))
for i in 1:8
    dense[:,i].=[1,2,3,4,5,6,7,8]
end
BBB=BandedBlockBandedMatrix(dense, [4, 4] ,[4, 4], (1, 1), (1, 1))
rowind,colind=findstructralnz(BBB)
@test [BBB[rowind[i],colind[i]] for i in 1:length(rowind)]==
    [1,2,3,1,2,3,4,2,3,4,5,6,7,5,6,7,8,6,7,8,
     1,2,3,1,2,3,4,2,3,4,5,6,7,5,6,7,8,6,7,8]

@testset "setindex" begin
    @testset "$(typeof(x))" for x in [
        zeros(3),
        falses(3),
        spzeros(3),
    ]
        y = setindex(x, true, 1)
        @test iszero(x)  # x is not mutated
        @test y[1] == true
        @test iszero(x[CartesianIndices(size(x)) .== [CartesianIndex(1)]])

        y2 = setindex(x, one.(x), :)
        @test iszero(x)
        @test all(isone, y2)
    end

    @testset "$(typeof(x))" for x in [
        zeros(3, 3),
        falses(3, 3),
        spzeros(3, 3),
    ]
        y = setindex(x, true, 1, 1)
        @test iszero(x)  # x is not mutated
        @test y[1, 1] == true
        @test iszero(x[CartesianIndices(size(x)) .== [CartesianIndex(1, 1)]])

        y2 = setindex(x, one.(x), :, :)
        @test iszero(x)
        @test all(isone, y2)
    end

    @testset "$(typeof(x))" for x in [
        zeros(3, 3, 3),
        falses(3, 3, 3),
    ]
        y = setindex(x, true, 1, 1, 1)
        @test iszero(x)  # x is not mutated
        @test y[1, 1, 1] == true
        @test iszero(x[CartesianIndices(size(x)) .== [CartesianIndex(1, 1, 1)]])

        y2 = setindex(x, one.(x), :, :, :)
        @test iszero(x)
        @test all(isone, y2)
    end
end

using SuiteSparse
@testset "lu_instance" begin
  for A in [
    randn(5, 5),
    @SMatrix(randn(5, 5)),
    @MMatrix(randn(5, 5)),
    sprand(50, 50, 0.5)
  ]
    @test lu_instance(A) isa typeof(lu(A))
  end
  @test lu_instance(1) === 1
end

using Random
using ArrayInterface: issingular
@testset "issingular" begin
    for T in [Float64, ComplexF64]
        R = randn(MersenneTwister(2), T, 5, 5)
        S = Symmetric(R)
        L = UpperTriangular(R)
        U = LowerTriangular(R)
        @test all(!issingular, [R, S, L, U, U'])
        R[:, 2] .= 0
        @test all(issingular, [R, L, U, U'])
        @test !issingular(S)
        R[2, :] .= 0
        @test issingular(S)
        @test all(!issingular, [UnitLowerTriangular(R), UnitUpperTriangular(R), UnitUpperTriangular(R)'])
    end
end

using ArrayInterface: zeromatrix
@test zeromatrix(rand(4,4,4)) == zeros(4*4*4,4*4*4)

using ArrayInterface: parent_type
@testset "Parent Type" begin
    x = ones(4, 4)
    @test parent_type(view(x, 1:2, 1:2)) <: typeof(x)
    @test parent_type(reshape(x, 2, :)) <: typeof(x)
    @test parent_type(transpose(x)) <: typeof(x)
    @test parent_type(Symmetric(x)) <: typeof(x)
    @test parent_type(UpperTriangular(x)) <: typeof(x)
end

@testset "Range Interface" begin
    @test isnothing(ArrayInterface.known_first(typeof(1:4)))
    @test isone(ArrayInterface.known_first(Base.OneTo(4)))
    @test isone(ArrayInterface.known_first(typeof(Base.OneTo(4))))

    @test isnothing(ArrayInterface.known_last(1:4))
    @test isnothing(ArrayInterface.known_last(typeof(1:4)))

    @test isnothing(ArrayInterface.known_step(typeof(1:0.2:4)))
    @test isone(ArrayInterface.known_step(1:4))
    @test isone(ArrayInterface.known_step(typeof(1:4)))
end

@testset "Memory Layout" begin
    A = rand(3,4,5)
    @test is_cpu_column_major(A)
    @test !is_cpu_column_major((1,2,3))
    @test is_cpu_column_major(view(A, 1, :, 2:4))
    @test !is_cpu_column_major(view(A, 1, :, 2:4)')
    @test !is_cpu_column_major(@SArray(rand(2,2,2)))
    @test is_cpu_column_major(@MArray(rand(2,2,2)))

    @test stridelayout(@SArray(rand(2,2,2))) == (1, 1, (1,2,3))
    @test stridelayout(A) == (1, 1, (1,2,3))
    @test stridelayout(PermutedDimsArray(A,(3,1,2))) == (2, 1, (3, 1, 2))
    @test stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])) == (1, 1, (1, 2))
    @test stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])') == (2, 1, (2, 1))
    @test stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:])) == (2, 1, (3, 1, 2))
    @test stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])) == (-1, 1, (2, 1))
    @test stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])') == (-1, 1, (1, 2))
    @test stridelayout(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])') == (1, 1, (1, 2))

    B = Array{Int8}(undef, 2,2,2,2);
    doubleperm = PermutedDimsArray(PermutedDimsArray(B,(4,2,3,1)), (4,2,1,3));
    @test collect(strides(B))[collect(last(stridelayout(doubleperm)))] == collect(strides(doubleperm))
end

@test ArrayInterface.can_avx(ArrayInterface.can_avx) == false

@testset "can_change_size" begin
    @test ArrayInterface.can_change_size([1])
    @test ArrayInterface.can_change_size(Vector{Int})
    @test ArrayInterface.can_change_size(Dict{Symbol,Any})
    @test !ArrayInterface.can_change_size(Base.ImmutableDict{Symbol,Int64})
    @test !ArrayInterface.can_change_size(Tuple{})
end

