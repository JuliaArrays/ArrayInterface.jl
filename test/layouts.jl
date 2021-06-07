
a = Array{Int}(undef, 5, 2, 5, 2); 
a[:] .= 1:100;
av = view(a, :, 1, :, :);
aperm = PermutedDimsArray(a, (3,1,4,2));
mv = view(aperm, :, 1, :, 1);
mvp = mv';

function layout_tests(x)
    la = ArrayInterface.LinearAccess()
    ca = ArrayInterface.CartesianAccess()

    lytla = @inferred(ArrayInterface.layout(x, la))
    lytca = @inferred(ArrayInterface.layout(x, ca))

    m = ArrayInterface.refdata(x)

    for i in eachindex(IndexLinear(), x)
        @test x[i] === lytla[i] === m[lytla[i]]
    end

    for i in eachindex(IndexCartesian(), x)
        @test x[i] === lytca[i] === m[lytca[i]]
    end
end

layout_tests(a)
layout_tests(av)
layout_tests(aperm)
layout_tests(mv)
layout_tests(mvp)

#=
ArrayInterface.getindex(aperm, :, 1, :, 1)

@btime ArrayInterface.StrideIndex(aperm)

ArrayInterface.layout(av, la)
=#
