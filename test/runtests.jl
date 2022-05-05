using ArrayInterface
using ArrayInterfaceCore
import ArrayInterfaceCore: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance,
    device, contiguous_axis, contiguous_batch_size, stride_rank, dense_dims,
    is_lazy_conjugate, parent_type, dimnames, zeromatrix
using BandedMatrices
using BlockBandedMatrices
using Base: setindex
using LinearAlgebra
using OffsetArrays
using Random
using SparseArrays
using Static
using StaticArrays
using SuiteSparse
using Test

if VERSION ≥ v"1.6"
    using Aqua
    Aqua.test_all(ArrayInterface)
end

@test isone(ArrayInterface.known_first(typeof(StaticArrays.SOneTo(7))))
@test ArrayInterface.known_last(typeof(StaticArrays.SOneTo(7))) == 7
@test ArrayInterface.known_length(typeof(StaticArrays.SOneTo(7))) == 7

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

@testset "parent_type" begin
    @test parent_type(SizedVector{1, Int, Vector{Int}}) <: Vector{Int}
end

@testset "Reinterpreted reshaped views" begin
    u_base = randn(1, 4, 4, 5);
    u_vectors = reshape(reinterpret(SVector{1, eltype(u_base)}, u_base),
                        Base.tail(size(u_base))...)
    u_view = view(u_vectors, 2, :, 3)
    u_view_reinterpreted = reinterpret(eltype(u_base), u_view)
    u_view_reshaped = reshape(u_view_reinterpreted, 1, length(u_view))

    # See https://github.com/JuliaArrays/ArrayInterface.jl/issues/163
    @test @inferred(ArrayInterface.strides(u_base)) == (StaticInt(1), 1, 4, 16)
    @test @inferred(ArrayInterface.strides(u_vectors)) == (StaticInt(1), 4, 16)
    @test @inferred(ArrayInterface.strides(u_view)) == (4,)
    @test @inferred(ArrayInterface.strides(u_view_reinterpreted)) == (4,)
    @test @inferred(ArrayInterface.strides(u_view_reshaped)) == (4, 4)
    @test @inferred(ArrayInterface.axes(u_vectors)) isa ArrayInterface.axes_types(u_vectors)
end

@testset "known_length" begin
    @test ArrayInterface.known_length(@inferred(ArrayInterface.indices(SOneTo(7)))) == 7

    x = view(SArray{Tuple{3,3,3}}(ones(3,3,3)), :, SOneTo(2), 2)
    @test @inferred(ArrayInterface.known_length(x)) == 6
    @test @inferred(ArrayInterface.known_length(x')) == 6

    v = @SVector rand(8);
    A = @MMatrix rand(7,6);
    T = SizedArray{Tuple{5,4,3}}(zeros(5,4,3));
    @test @inferred(ArrayInterface.length(v)) === StaticInt(8)
    @test @inferred(ArrayInterface.length(A)) === StaticInt(42)
    @test @inferred(ArrayInterface.length(T)) === StaticInt(60)
end

@testset "indices" begin
    A23 = ones(2,3); SA23 = @SMatrix ones(2,3);
    A32 = ones(3,2); SA32 = @SMatrix ones(3,2);

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
    Am = @MMatrix rand(2,10);
    @test @inferred(ArrayInterface.strides(view(Am,1,:))) === (StaticInt(2),)


    O = OffsetArray(A, 3, 7, 10);
    Op = PermutedDimsArray(O,(3,1,2));
    @test @inferred(ArrayInterface.offsets(O)) === (4, 8, 11)
    @test @inferred(ArrayInterface.offsets(Op)) === (11, 4, 8)

    @test @inferred(ArrayInterface.offsets((1,2,3))) === (StaticInt(1),)
    o = OffsetArray(vec(A), 8);
    @test @inferred(ArrayInterface.offset1(o)) === 9

    @test @inferred(device(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173))) === ArrayInterface.CPUPointer()
    @test @inferred(device(view(OffsetArray(A,2,3,-12), 4, :, -11:-9))) === ArrayInterface.CPUPointer()
    @test @inferred(device(view(OffsetArray(A,2,3,-12), 3, :, [-11,-10,-9])')) === ArrayInterface.CPUIndex()
 
    @test @inferred(ArrayInterface.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),1)) === Base.Slice(ArrayInterface.OptionallyStaticUnitRange(4,6))
    @test @inferred(ArrayInterface.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),2)) === Base.Slice(ArrayInterface.OptionallyStaticUnitRange(-172,-170))

    @test @inferred(contiguous_axis(@SArray(zeros(2,2,2)))) === ArrayInterface.StaticInt(1)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@SArray(zeros(2,2,2)))) == (true,false,false)
    @test @inferred(contiguous_batch_size(@SArray(zeros(2,2,2)))) === ArrayInterface.StaticInt(0)
    @test @inferred(stride_rank(@SArray(zeros(2,2,2)))) == (1, 2, 3)
    @test @inferred(ArrayInterface.is_column_major(@SArray(zeros(2,2,2)))) === True()
    @test @inferred(dense_dims(@SArray(zeros(2,2,2)))) == (true,true,true)
    @test @inferred(device(OffsetArray(@SArray(zeros(2,2,2)),-123,29,3231))) === ArrayInterface.CPUTuple()
    @test @inferred(device(OffsetArray(@view(@SArray(zeros(2,2,2))[1,1:2,:]),-3,4))) === ArrayInterface.CPUTuple()
    @test @inferred(device(OffsetArray(@MArray(zeros(2,2,2)),8,-2,-5))) === ArrayInterface.CPUPointer()

    x = @SVector [1,2,3]
    @test ArrayInterfaceCore.ismutable(x) == false
    @test ArrayInterfaceCore.ismutable(view(x, 1:2)) == false
    x = @MVector [1,2,3]
    @test ArrayInterfaceCore.ismutable(x) == true
    @test ArrayInterfaceCore.ismutable(view(x, 1:2)) == true

