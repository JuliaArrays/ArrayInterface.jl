using ArrayInterface, Test
using Base: setindex
import ArrayInterface: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance, device, contiguous_axis, contiguous_batch_size, stride_rank, dense_dims
@test ArrayInterface.ismutable(rand(3))

using StaticArrays
@test ArrayInterface.ismutable(@SVector [1,2,3]) == false
@test ArrayInterface.ismutable(@MVector [1,2,3]) == true
@test ArrayInterface.ismutable(1:10) == false
@test ArrayInterface.ismutable((0.1,1.0)) == false
@test isone(ArrayInterface.known_first(typeof(StaticArrays.SOneTo(7))))
@test ArrayInterface.known_last(typeof(StaticArrays.SOneTo(7))) == 7
@test ArrayInterface.known_length(typeof(StaticArrays.SOneTo(7))) == 7

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
    @test parent_type(PermutedDimsArray(x, (2,1))) <: typeof(x)
    @test parent_type(Base.Slice(1:10)) <: UnitRange{Int}
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
    A = zeros(3,4,5);
    @test device(A) === ArrayInterface.CPUPointer()
    @test device((1,2,3)) === ArrayInterface.CPUIndex()
    @test device(PermutedDimsArray(A,(3,1,2))) === ArrayInterface.CPUPointer()
    @test device(view(A, 1, :, 2:4)) === ArrayInterface.CPUPointer()
    @test device(view(A, 1, :, 2:4)') === ArrayInterface.CPUPointer()
    @test device(@SArray(zeros(2,2,2))) === ArrayInterface.CPUIndex()
    @test device(@view(@SArray(zeros(2,2,2))[1,1:2,:])) === ArrayInterface.CPUIndex()
    @test device(@MArray(zeros(2,2,2))) === ArrayInterface.CPUPointer()
    @test isnothing(device("Hello, world!"))

    @test @inferred(contiguous_axis(@SArray(zeros(2,2,2)))) === ArrayInterface.Contiguous(1)
    @test @inferred(contiguous_axis(A)) === ArrayInterface.Contiguous(1)
    @test @inferred(contiguous_axis(PermutedDimsArray(A,(3,1,2)))) === ArrayInterface.Contiguous(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterface.Contiguous(1)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === ArrayInterface.Contiguous(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterface.Contiguous(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterface.Contiguous(-1)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterface.Contiguous(-1)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterface.Contiguous(1)

    @test @inferred(ArrayInterface.contiguous_axis_indicator(@SArray(zeros(2,2,2)))) === (Val(true),Val(false),Val(false))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(A)) === (Val(true),Val(false),Val(false))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(PermutedDimsArray(A,(3,1,2)))) === (Val(false),Val(true),Val(false))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === (Val(true),Val(false))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === (Val(false),Val(true))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === (Val(false),Val(true),Val(false))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === (Val(false),Val(false))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === (Val(false),Val(false))
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === (Val(true),Val(false))

    @test @inferred(contiguous_batch_size(@SArray(zeros(2,2,2)))) === ArrayInterface.ContiguousBatch(0)
    @test @inferred(contiguous_batch_size(A)) === ArrayInterface.ContiguousBatch(0)
    @test @inferred(contiguous_batch_size(PermutedDimsArray(A,(3,1,2)))) === ArrayInterface.ContiguousBatch(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterface.ContiguousBatch(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === ArrayInterface.ContiguousBatch(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterface.ContiguousBatch(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterface.ContiguousBatch(-1)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterface.ContiguousBatch(-1)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterface.ContiguousBatch(0)

    @test @inferred(stride_rank(@SArray(zeros(2,2,2)))) === ArrayInterface.StrideRank((1, 2, 3))
    @test @inferred(stride_rank(A)) === ArrayInterface.StrideRank((1,2,3))
    @test @inferred(stride_rank(PermutedDimsArray(A,(3,1,2)))) === ArrayInterface.StrideRank((3, 1, 2))
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterface.StrideRank((1, 2))
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === ArrayInterface.StrideRank((2, 1))
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterface.StrideRank((3, 1, 2))
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterface.StrideRank((3, 2))
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterface.StrideRank((2, 3))
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterface.StrideRank((1, 3))

    @test @inferred(dense_dims(@SArray(zeros(2,2,2)))) === ArrayInterface.DenseDims((true,true,true))
    @test @inferred(dense_dims(A)) === ArrayInterface.DenseDims((true,true,true))
    @test @inferred(dense_dims(PermutedDimsArray(A,(3,1,2)))) === ArrayInterface.DenseDims((true,true,true))
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterface.DenseDims((true,false))
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === ArrayInterface.DenseDims((false,true))
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterface.DenseDims((false,true,false))
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,:,1:2]))) === ArrayInterface.DenseDims((false,true,true))
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterface.DenseDims((false,false))
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterface.DenseDims((false,false))
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterface.DenseDims((true,false))

    B = Array{Int8}(undef, 2,2,2,2);
    doubleperm = PermutedDimsArray(PermutedDimsArray(B,(4,2,3,1)), (4,2,1,3));
    @test collect(strides(B))[collect(stride_rank(doubleperm))] == collect(strides(doubleperm))
end

@testset "Static-Dynamic Size and Strides" begin
    A = zeros(3,4,5); Ap = @view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])';
    S = @SArray zeros(2,3,4); Sp = @view(PermutedDimsArray(S,(3,1,2))[2:3,1:2,:]);
    M = @MArray zeros(2,3,4); Mp = @view(PermutedDimsArray(M,(3,1,2))[:,2,:])';
    Sp2 = @view(PermutedDimsArray(S,(3,2,1))[2:3,:,:]);
    Mp2 = @view(PermutedDimsArray(M,(3,1,2))[2:3,:,2])';
    
    @test @inferred(ArrayInterface.sdsize(A)) === ArrayInterface.SDTuple{(-1,-1,-1)}((3,4,5))
    @test @inferred(ArrayInterface.sdsize(Ap)) === ArrayInterface.SDTuple{(-1,-1)}((2,5))
    @test @inferred(Tuple(ArrayInterface.sdsize(A))) === size(A)
    @test @inferred(Tuple(ArrayInterface.sdsize(Ap))) === size(Ap)
    
    @test @inferred(ArrayInterface.sdsize(S)) === ArrayInterface.SDTuple{(2, 3, 4)}(())
    @test @inferred(ArrayInterface.sdsize(Sp)) === ArrayInterface.SDTuple{(-1, -1, 3)}((2, 2))
    @test @inferred(ArrayInterface.sdsize(Sp2)) === ArrayInterface.SDTuple{(-1, 3, 2)}((2, ))
    @test @inferred(Tuple(ArrayInterface.sdsize(S))) === size(S)
    @test @inferred(Tuple(ArrayInterface.sdsize(Sp))) === size(Sp)
    @test @inferred(Tuple(ArrayInterface.sdsize(Sp2))) === size(Sp2)
    
    @test @inferred(ArrayInterface.sdsize(M)) === ArrayInterface.SDTuple{(2,3,4)}(())
    @test @inferred(ArrayInterface.sdsize(Mp)) === ArrayInterface.SDTuple{(3,4)}(())
    @test @inferred(ArrayInterface.sdsize(Mp2)) === ArrayInterface.SDTuple{(2,-1)}((2,))
    @test @inferred(Tuple(ArrayInterface.sdsize(M))) === size(M)
    @test @inferred(Tuple(ArrayInterface.sdsize(Mp))) === size(Mp)
    @test @inferred(Tuple(ArrayInterface.sdsize(Mp2))) === size(Mp2)

    @test @inferred(ArrayInterface.sdstrides(A)) === ArrayInterface.SDTuple{(1,-1,-1)}((3,12))
    @test @inferred(ArrayInterface.sdstrides(Ap)) === ArrayInterface.SDTuple{(1,-1)}((12,))
    @test @inferred(Tuple(ArrayInterface.sdstrides(A))) === strides(A)
    @test @inferred(Tuple(ArrayInterface.sdstrides(Ap))) === strides(Ap)
    
    @test @inferred(ArrayInterface.sdstrides(S)) === ArrayInterface.SDTuple{(1,2,6)}(())
    @test @inferred(ArrayInterface.sdstrides(Sp)) === ArrayInterface.SDTuple{(6,1,2)}(())
    @test @inferred(ArrayInterface.sdstrides(Sp2)) === ArrayInterface.SDTuple{(6, 2, 1)}(())
    
    @test @inferred(ArrayInterface.sdstrides(M)) === ArrayInterface.SDTuple{(1,2,6)}(())
    @test @inferred(ArrayInterface.sdstrides(Mp)) === ArrayInterface.SDTuple{(2,6)}(())
    @test @inferred(ArrayInterface.sdstrides(Mp2)) === ArrayInterface.SDTuple{(1,6)}(())
    @test @inferred(Tuple(ArrayInterface.sdstrides(M))) === strides(M)
    @test @inferred(Tuple(ArrayInterface.sdstrides(Mp))) === strides(Mp)
    @test @inferred(Tuple(ArrayInterface.sdstrides(Mp2))) === strides(Mp2)

    @test isnothing(ArrayInterface.sdsize((1,2,3)))
    @test isnothing(ArrayInterface.sdstrides((1,2,3)))
end

@test ArrayInterface.can_avx(ArrayInterface.can_avx) == false

@testset "can_change_size" begin
    @test ArrayInterface.can_change_size([1])
    @test ArrayInterface.can_change_size(Vector{Int})
    @test ArrayInterface.can_change_size(Dict{Symbol,Any})
    @test !ArrayInterface.can_change_size(Base.ImmutableDict{Symbol,Int64})
    @test !ArrayInterface.can_change_size(Tuple{})
end

@testset "known_length" begin
    @test ArrayInterface.known_length(ArrayInterface.indices(SOneTo(7))) == 7
    @test ArrayInterface.known_length(1:2) == nothing
    @test ArrayInterface.known_length((1,)) == 1
    @test ArrayInterface.known_length((a=1,b=2)) == 2
end

@testset "indices" begin
    @test @inferred(ArrayInterface.indices((ones(2, 3), ones(3, 2)))) == 1:6
    @test @inferred(ArrayInterface.indices(ones(2, 3))) == 1:6
    @test @inferred(ArrayInterface.indices(ones(2, 3), 1)) == 1:2
    @test @inferred(ArrayInterface.indices((ones(2, 3), ones(3, 2)), (1, 2))) == 1:2
    @test @inferred(ArrayInterface.indices((ones(2, 3), ones(2, 3)), 1)) == 1:2
    @test_throws AssertionError ArrayInterface.indices((ones(2, 3), ones(3, 3)), 1)
    @test_throws AssertionError ArrayInterface.indices((ones(2, 3), ones(3, 3)), (1, 2))
end

