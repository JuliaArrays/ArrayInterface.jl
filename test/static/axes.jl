

v = Array{Float64}(undef, 4)
m = Array{Float64}(undef, 4, 3)
@test @inferred(ArrayInterface.static_axes(v')) === (StaticInt(1):StaticInt(1),Base.OneTo(4))
@test @inferred(ArrayInterface.static_axes(m')) === (Base.OneTo(3),Base.OneTo(4))
@test ArrayInterface.static_axes(v', StaticInt(1)) === StaticInt(1):StaticInt(1)
@test ArrayInterface.static_axes(v, StaticInt(2)) === StaticInt(1):StaticInt(1)
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
    @test @inferred(ArrayInterface.parent_type(ArrayInterface.LazyAxis{4}(SA))) <: Static.SOneTo{1}
    @test @inferred(ArrayInterface.parent_type(ArrayInterface.LazyAxis{:}(SA))) <: Static.SOneTo{60}
    @test @inferred(IndexStyle(SA)) isa IndexLinear
    @test @inferred(IndexStyle(DA)) isa IndexLinear
    @test ArrayInterface.can_change_size(ArrayInterface.LazyAxis{1,Vector{Any}})

    Aperm = PermutedDimsArray(A, (3,1,2))
    Aview = @view(Aperm[:,1:2,1])
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aperm))) === @inferred(ArrayInterface.static_axes(Aperm))
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aview))) === @inferred(ArrayInterface.static_axes(Aview))
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aview'))) === @inferred(ArrayInterface.static_axes(Aview'))
    @test map(parent, @inferred(ArrayInterface.lazy_axes((1:2)'))) === @inferred(ArrayInterface.static_axes((1:2)'))

    @test_throws DimensionMismatch ArrayInterface.LazyAxis{0}(A)
end

@testset "`axes(A, dim)`` with `dim > ndims(A)` (#224)" begin
    m = 2
    n = 3
    B = Array{Float64, 2}(undef, m, n)
    b = view(B, :, 1)

    @test @inferred(ArrayInterface.static_axes(B, 1)) == 1:m
    @test @inferred(ArrayInterface.static_axes(B, 2)) == 1:n
    @test @inferred(ArrayInterface.static_axes(B, 3)) == 1:1

    @test @inferred(ArrayInterface.static_axes(B, static(1))) == 1:m
    @test @inferred(ArrayInterface.static_axes(B, static(2))) == 1:n
    @test @inferred(ArrayInterface.static_axes(B, static(3))) == 1:1

    @test @inferred(ArrayInterface.static_axes(b, 1)) == 1:m
    @test @inferred(ArrayInterface.static_axes(b, 2)) == 1:1
    @test @inferred(ArrayInterface.static_axes(b, 3)) == 1:1

    @test @inferred(ArrayInterface.static_axes(b, static(1))) == 1:m
    @test @inferred(ArrayInterface.static_axes(b, static(2))) == 1:1
    @test @inferred(ArrayInterface.static_axes(b, static(3))) == 1:1

    # multidimensional subindices
    vx = view(rand(4), reshape(1:4, 2, 2))
    @test @inferred(axes(vx)) == (1:2, 1:2)
end

@testset "SubArray Adjoint Axis" begin
  N = 4; d = rand(N);

  @test @inferred(ArrayInterface.axes_types(typeof(view(d',:,1:2)))) ===
    Tuple{Static.OptionallyStaticUnitRange{StaticInt{1}, StaticInt{1}}, Base.OneTo{Int64}} ===
    typeof(@inferred(ArrayInterface.static_axes(view(d',:,1:2)))) ===
    typeof((ArrayInterface.static_axes(view(d',:,1:2),1),ArrayInterface.static_axes(view(d',:,1:2),2)))
end
if isdefined(Base, :ReshapedReinterpretArray)
  @testset "ReshapedReinterpretArray" begin
    a = rand(3, 5)
    ua = reinterpret(reshape, UInt64, a)
    @test ArrayInterface.static_axes(ua) === ArrayInterface.static_axes(a)
    @test ArrayInterface.static_axes(ua, 1) === ArrayInterface.static_axes(a, 1)
    @test @inferred(ArrayInterface.static_axes(ua)) isa ArrayInterface.axes_types(ua)
    u8a = reinterpret(reshape, UInt8, a)
    @test @inferred(ArrayInterface.static_axes(u8a)) isa ArrayInterface.axes_types(u8a)
    @test @inferred(ArrayInterface.static_axes(u8a, static(1))) isa ArrayInterface.axes_types(u8a, 1)
    @test @inferred(ArrayInterface.static_axes(u8a, static(2))) isa ArrayInterface.axes_types(u8a, 2)
    fa = reinterpret(reshape, Float64, copy(u8a))
    @inferred(ArrayInterface.static_axes(fa)) isa ArrayInterface.axes_types(fa)
  end
end
