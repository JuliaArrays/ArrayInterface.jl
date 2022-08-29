

v = Array{Float64}(undef, 4)
m = Array{Float64}(undef, 4, 3)
@test @inferred(ArrayInterface.axes(v')) === (StaticInt(1):StaticInt(1),Base.OneTo(4))
@test @inferred(ArrayInterface.axes(m')) === (Base.OneTo(3),Base.OneTo(4))
@test ArrayInterface.axes(v', StaticInt(1)) === StaticInt(1):StaticInt(1)
@test ArrayInterface.axes(v, StaticInt(2)) === StaticInt(1):StaticInt(1)
@test ArrayInterface.axes_types(view(CartesianIndices(map(Base.Slice, (0:3, 3:5))), 0, :), 1) <: Base.IdentityUnitRange

@testset "LazyAxis" begin
    A = zeros(3,4,5);
    SA = MArray(zeros(3,4,5))
    DA = MArray(zeros(3,4,5), LinearIndices((Base.Slice(1:3), 1:4, 1:5)))
    lz1 = ArrayInterface.LazyAxis{1}(A)
    slz1 = ArrayInterface.LazyAxis{1}(SA)
    dlz1 = ArrayInterface.LazyAxis{1}(DA)
    lzc = ArrayInterface.LazyAxis{:}(A)
    slzc = ArrayInterface.LazyAxis{:}(SA)
    dlzc = ArrayInterface.LazyAxis{:}(DA)

    @test @inferred(first(lz1)) === @inferred(first(slz1)) === @inferred(first(dlz1))
    @test @inferred(first(lzc)) === @inferred(first(slzc)) === @inferred(first(dlzc))
    @test @inferred(last(lz1)) === @inferred(last(slz1)) === @inferred(last(dlz1))
    @test @inferred(last(lzc)) === @inferred(last(slzc)) === @inferred(last(dlzc))
    @test @inferred(length(lz1)) === @inferred(length(slz1)) === @inferred(length(dlz1))
    @test @inferred(length(lzc)) === @inferred(length(slzc)) === @inferred(length(dlzc))
    @test @inferred(Base.to_shape(lzc)) == length(slzc) == length(dlzc)
    @test @inferred(Base.checkindex(Bool, lzc, 1)) & @inferred(Base.checkindex(Bool, slzc, 1))
    @test axes(lzc)[1] == Base.axes1(lzc) == axes(Base.Slice(lzc))[1] == Base.axes1(Base.Slice(lzc))
    @test keys(axes(A, 1)) == @inferred(keys(lz1))

    @test @inferred(ArrayInterface.known_first(slzc)) === 1
    @test @inferred(ArrayInterface.known_length(slz1)) === 3

    @test @inferred(getindex(lz1, 2)) == 2
    @test @inferred(getindex(lz1, 1:2)) == 1:2
    @test @inferred(getindex(lz1, 1:1:3)) == 1:1:3

    @test @inferred(ArrayInterface.parent_type(ArrayInterface.LazyAxis{:}(A))) <: Base.OneTo{Int}
    @test @inferred(ArrayInterface.parent_type(ArrayInterface.LazyAxis{4}(SA))) <: ArrayInterface.SOneTo{1}
    @test @inferred(ArrayInterface.parent_type(ArrayInterface.LazyAxis{:}(SA))) <: ArrayInterface.SOneTo{60}
    @test @inferred(IndexStyle(SA)) isa IndexLinear
    @test @inferred(IndexStyle(DA)) isa IndexLinear
    @test ArrayInterface.can_change_size(ArrayInterface.LazyAxis{1,Vector{Any}})

    Aperm = PermutedDimsArray(A, (3,1,2))
    Aview = @view(Aperm[:,1:2,1])
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aperm))) === @inferred(ArrayInterface.axes(Aperm))
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aview))) === @inferred(ArrayInterface.axes(Aview))
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aview'))) === @inferred(ArrayInterface.axes(Aview'))
    @test map(parent, @inferred(ArrayInterface.lazy_axes((1:2)'))) === @inferred(ArrayInterface.axes((1:2)'))

    @test_throws DimensionMismatch ArrayInterface.LazyAxis{0}(A)
end

@testset "`axes(A, dim)`` with `dim > ndims(A)` (#224)" begin
    m = 2
    n = 3
    B = Array{Float64, 2}(undef, m, n)
    b = view(B, :, 1)

    @test @inferred(ArrayInterface.axes(B, 1)) == 1:m
    @test @inferred(ArrayInterface.axes(B, 2)) == 1:n
    @test @inferred(ArrayInterface.axes(B, 3)) == 1:1

    @test @inferred(ArrayInterface.axes(B, static(1))) == 1:m
    @test @inferred(ArrayInterface.axes(B, static(2))) == 1:n
    @test @inferred(ArrayInterface.axes(B, static(3))) == 1:1

    @test @inferred(ArrayInterface.axes(b, 1)) == 1:m
    @test @inferred(ArrayInterface.axes(b, 2)) == 1:1
    @test @inferred(ArrayInterface.axes(b, 3)) == 1:1

    @test @inferred(ArrayInterface.axes(b, static(1))) == 1:m
    @test @inferred(ArrayInterface.axes(b, static(2))) == 1:1
    @test @inferred(ArrayInterface.axes(b, static(3))) == 1:1

    # multidimensional subindices
    vx = view(rand(4), reshape(1:4, 2, 2))
    @test @inferred(axes(vx)) == (1:2, 1:2)
end

@testset "SubArray Adjoint Axis" begin
  N = 4; d = rand(N);

  @test @inferred(ArrayInterface.axes_types(typeof(view(d',:,1:2)))) ===
    Tuple{ArrayInterface.OptionallyStaticUnitRange{StaticInt{1}, StaticInt{1}}, Base.OneTo{Int64}} ===
    typeof(@inferred(ArrayInterface.axes(view(d',:,1:2)))) ===
    typeof((ArrayInterface.axes(view(d',:,1:2),1),ArrayInterface.axes(view(d',:,1:2),2)))
end
if isdefined(Base, :ReshapedReinterpretArray)
  @testset "ReshapedReinterpretArray" begin
    a = rand(3, 5)
    ua = reinterpret(reshape, UInt64, a)
    @test ArrayInterface.axes(ua) === ArrayInterface.axes(a)
    @test ArrayInterface.axes(ua, 1) === ArrayInterface.axes(a, 1)
    @test @inferred(ArrayInterface.axes(ua)) isa ArrayInterface.axes_types(ua)
    u8a = reinterpret(reshape, UInt8, a)
    @test @inferred(ArrayInterface.axes(u8a)) isa ArrayInterface.axes_types(u8a)
    @test @inferred(ArrayInterface.axes(u8a, static(1))) isa ArrayInterface.axes_types(u8a, 1)
    @test @inferred(ArrayInterface.axes(u8a, static(2))) isa ArrayInterface.axes_types(u8a, 2)
    fa = reinterpret(reshape, Float64, copy(u8a))
    @inferred(ArrayInterface.axes(fa)) isa ArrayInterface.axes_types(fa)
  end
end

@testset "index_labels" begin
    colors = LabelledArray([(R = rand(), G = rand(), B = rand()) for i ∈ 1:100], (range(-10, 10, length=100),));
    caxis = ArrayInterface.LazyAxis{1}(colors);
    colormat = reinterpret(reshape, Float64, colors);
    cmat_view1 = view(colormat, :, 4);
    cmat_view2 = view(colormat, :, 4:7);
    cmat_view3 = view(colormat, 2:3,:);
    absym_abstr = LabelledArray(rand(Int64, 2,2), ([:a, :b], ["a", "b"],));

    @test @inferred(ArrayInterface.index_labels(colors)) == (range(-10, 10, length=100),)
    @test @inferred(ArrayInterface.index_labels(caxis)) == (range(-10, 10, length=100),)
    @test ArrayInterface.index_labels(view(colors, :, :), 2) === nothing
    @test @inferred(ArrayInterface.index_labels(LinearIndices((caxis,)))) == (range(-10, 10, length=100),)
    @test @inferred(ArrayInterface.index_labels(colormat)) == ((:R, :G, :B), range(-10, 10, length=100))
    @test @inferred(ArrayInterface.index_labels(colormat')) == (range(-10, 10, length=100), (:R, :G, :B))
    @test @inferred(ArrayInterface.index_labels(cmat_view1)) == ((:R, :G, :B),)
    @test @inferred((ArrayInterface.index_labels(cmat_view2))) == ((:R, :G, :B), -9.393939393939394:0.20202020202020202:-8.787878787878787)
    @test @inferred((ArrayInterface.index_labels(view(colormat, 1, :)'))) == (nothing, range(-10, 10, length=100))
    # can't infer this b/c tuple is being indexed by range
    @test ArrayInterface.index_labels(cmat_view3) == ((:G, :B), -10.0:0.20202020202020202:10.0)
    @test @inferred(ArrayInterface.index_labels(Symmetric(view(colormat, :, 1:3)))) == ((:R, :G, :B), -10.0:0.20202020202020202:-9.595959595959595)

    @test @inferred(ArrayInterface.index_labels(reinterpret(Int8, absym_abstr))) == (nothing, ["a", "b"])
    @test @inferred(ArrayInterface.index_labels(reinterpret(reshape, Int8, absym_abstr))) == (nothing, [:a, :b], ["a", "b"])
    @test @inferred(ArrayInterface.index_labels(reinterpret(reshape, Int64, LabelledArray(rand(Int32, 2,2), ([:a, :b], ["a", "b"],))))) == (["a", "b"],)
    @test @inferred(ArrayInterface.index_labels(reinterpret(reshape, Float64, LabelledArray(rand(Int64, 2,2), ([:a, :b], ["a", "b"],))))) == ([:a, :b], ["a", "b"],)
    @test @inferred(ArrayInterface.index_labels(reinterpret(Float64, absym_abstr))) == ([:a, :b], ["a", "b"],)

    @test ArrayInterface.has_index_labels(colors)
    @test ArrayInterface.has_index_labels(caxis)
    @test ArrayInterface.has_index_labels(colormat)
    @test ArrayInterface.has_index_labels(cmat_view1)
    @test !ArrayInterface.has_index_labels(view(colors, :, :))

    @test @inferred(ArrayInterface.getindex(colormat, :R, :)) == colormat[1, :]
    @test @inferred(ArrayInterface.getindex(cmat_view1, :R)) == cmat_view1[1]
    @test @inferred(ArrayInterface.getindex(colormat, :,ArrayInterface.IndexLabel(-9.595959595959595))) == colormat[:, 3]
    @test @inferred(ArrayInterface.getindex(colormat, :,<=(ArrayInterface.IndexLabel(-9.595959595959595)))) == colormat[:, 1:3]
    @test @inferred(ArrayInterface.getindex(absym_abstr, :, ["a"])) == absym_abstr[:,[1]]
end
