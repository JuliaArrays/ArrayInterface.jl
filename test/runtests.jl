using ArrayInterface
using ArrayInterfaceCore
import ArrayInterfaceCore: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance,
    device, contiguous_axis, contiguous_batch_size, stride_rank, dense_dims,
    is_lazy_conjugate, parent_type, dimnames, zeromatrix
using BandedMatrices
using BlockBandedMatrices
using Base: setindex
using LinearAlgebra
using Random
using OffsetArrays
using Static
using StaticArrays
using Test

@testset "StaticArrays" begin
    x = @SVector [1,2,3]
    @test ArrayInterfaceCore.ismutable(x) == false
    @test ArrayInterfaceCore.ismutable(view(x, 1:2)) == false
    @test ArrayInterfaceCore.can_setindex(typeof(x)) == false
    @test ArrayInterfaceCore.buffer(x) == x.data
    @test @inferred(ArrayInterfaceCore.device(typeof(x))) === ArrayInterfaceCore.CPUTuple()

    x = @MVector [1,2,3]
    @test ArrayInterfaceCore.ismutable(x) == true
    @test ArrayInterfaceCore.ismutable(view(x, 1:2)) == true
    @test @inferred(ArrayInterfaceCore.device(typeof(x))) === ArrayInterfaceCore.CPUPointer()

    A = @SMatrix(randn(5, 5))
    @test lu_instance(A) isa typeof(lu(A))
    A = @MMatrix(randn(5, 5))
    @test lu_instance(A) isa typeof(lu(A))
 
    @test isone(ArrayInterfaceCore.known_first(typeof(StaticArrays.SOneTo(7))))
    @test ArrayInterfaceCore.known_last(typeof(StaticArrays.SOneTo(7))) == 7
    @test ArrayInterfaceCore.known_length(typeof(StaticArrays.SOneTo(7))) == 7

    @test parent_type(SizedVector{1, Int, Vector{Int}}) <: Vector{Int}
    @test ArrayInterfaceCore.known_length(@inferred(ArrayInterfaceCore.indices(SOneTo(7)))) == 7

    x = view(SArray{Tuple{3,3,3}}(ones(3,3,3)), :, SOneTo(2), 2)
    @test @inferred(ArrayInterfaceCore.known_length(x)) == 6
    @test @inferred(ArrayInterfaceCore.known_length(x')) == 6

    v = @SVector rand(8);
    A = @MMatrix rand(7, 6);
    T = SizedArray{Tuple{5,4,3}}(zeros(5,4,3));
    @test @inferred(ArrayInterfaceCore.length(v)) === StaticInt(8)
    @test @inferred(ArrayInterfaceCore.length(A)) === StaticInt(42)
    @test @inferred(ArrayInterfaceCore.length(T)) === StaticInt(60)

    x = @SMatrix rand(Float32, 2, 2)
    y = @SVector rand(4)
    yr = ArrayInterfaceCore.restructure(x, y)
    @test yr isa SMatrix{2, 2}
    @test Base.size(yr) == (2,2)
    @test vec(yr) == vec(y)
    z = rand(4)
    zr = ArrayInterfaceCore.restructure(x, z)
    @test zr isa SMatrix{2, 2}
    @test Base.size(zr) == (2,2)
    @test vec(zr) == vec(z)

    Am = @MMatrix rand(2,10);
    @test @inferred(ArrayInterfaceCore.strides(view(Am,1,:))) === (StaticInt(2),)

    @test @inferred(contiguous_axis(@SArray(zeros(2,2,2)))) === ArrayInterfaceCore.StaticInt(1)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@SArray(zeros(2,2,2)))) == (true,false,false)
    @test @inferred(contiguous_batch_size(@SArray(zeros(2,2,2)))) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(stride_rank(@SArray(zeros(2,2,2)))) == (1, 2, 3)
    @test @inferred(ArrayInterfaceCore.is_column_major(@SArray(zeros(2,2,2)))) === True()
    @test @inferred(dense_dims(@SArray(zeros(2,2,2)))) == (true,true,true)
end

@testset "BandedMatrices" begin
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
    @test ArrayInterfaceCore.isstructured(typeof(B))
    @test ArrayInterfaceCore.fast_matrix_colors(typeof(B))
end

@testset "BlockBandedMatrices" begin
    BB=BlockBandedMatrix(Ones(10,10),[1,2,3,4],[4,3,2,1],(1,0))
    BB[Block(1,1)].=[1 2 3 4]
    BB[Block(2,1)].=[5 6 7 8;9 10 11 12]
    rowind,colind=findstructralnz(BB)
    @test [BB[rowind[i],colind[i]] for i in 1:length(rowind)]==
        [1,5,9,2,6,10,3,7,11,4,8,12,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1]
    @test ArrayInterfaceCore.isstructured(typeof(BB))
    @test has_sparsestruct(typeof(BB))
    @test ArrayInterfaceCore.fast_matrix_colors(typeof(BB))

    dense=collect(Ones(8,8))
    for i in 1:8
        dense[:,i].=[1,2,3,4,5,6,7,8]
    end
    BBB=BandedBlockBandedMatrix(dense, [4, 4] ,[4, 4], (1, 1), (1, 1))
    rowind,colind=findstructralnz(BBB)
    @test [BBB[rowind[i],colind[i]] for i in 1:length(rowind)]==
        [1,2,3,1,2,3,4,2,3,4,5,6,7,5,6,7,8,6,7,8,
         1,2,3,1,2,3,4,2,3,4,5,6,7,5,6,7,8,6,7,8]
    @test ArrayInterfaceCore.isstructured(typeof(BBB))
    @test has_sparsestruct(typeof(BBB))
    @test ArrayInterfaceCore.fast_matrix_colors(typeof(BBB))
end

@testset "OffsetArrays" begin
    A = zeros(3, 4, 5);
    O = OffsetArray(A, 3, 7, 10);
    Op = PermutedDimsArray(O,(3,1,2));
    @test @inferred(ArrayInterfaceCore.offsets(O)) === (4, 8, 11)
    @test @inferred(ArrayInterfaceCore.offsets(Op)) === (11, 4, 8)

    @test @inferred(ArrayInterfaceCore.offsets((1,2,3))) === (StaticInt(1),)
    o = OffsetArray(vec(A), 8);
    @test @inferred(ArrayInterfaceCore.offset1(o)) === 9

    @test @inferred(device(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173))) === ArrayInterfaceCore.CPUPointer()
    @test @inferred(device(view(OffsetArray(A,2,3,-12), 4, :, -11:-9))) === ArrayInterfaceCore.CPUPointer()
    @test @inferred(device(view(OffsetArray(A,2,3,-12), 3, :, [-11,-10,-9])')) === ArrayInterfaceCore.CPUIndex()

    @test @inferred(ArrayInterfaceCore.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),1)) === Base.Slice(ArrayInterfaceCore.OptionallyStaticUnitRange(4,6))
    @test @inferred(ArrayInterfaceCore.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),2)) === Base.Slice(ArrayInterfaceCore.OptionallyStaticUnitRange(-172,-170))

    @test @inferred(device(OffsetArray(@SArray(zeros(2,2,2)),-123,29,3231))) === ArrayInterfaceCore.CPUTuple()
    @test @inferred(device(OffsetArray(@view(@SArray(zeros(2,2,2))[1,1:2,:]),-3,4))) === ArrayInterfaceCore.CPUTuple()
    @test @inferred(device(OffsetArray(@MArray(zeros(2,2,2)),8,-2,-5))) === ArrayInterfaceCore.CPUPointer()
end

