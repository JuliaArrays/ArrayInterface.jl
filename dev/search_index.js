var documenterSearchIndex = {"docs":
[{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"CurrentModule = ArrayInterface","category":"page"},{"location":"#ArrayInterface","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface","text":"","category":"section"},{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"Designs for new Base array interface primitives, used widely through scientific machine learning (SciML) and other organizations","category":"page"},{"location":"#Extensions","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"Extensions","text":"","category":"section"},{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"ArrayInterface.jl uses extension packages in order to add support for popular libraries to its interface functions. These packages are:","category":"page"},{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"BandedMatrices.jl\nBlockBandedMatrices.jl\nGPUArrays.jl / CUDA.jl\nOffsetArrays.jl\nTracker.jl","category":"page"},{"location":"#StaticArrayInterface.jl","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"StaticArrayInterface.jl","text":"","category":"section"},{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"If one is looking for an interface which includes functionality for statically-computed values, see  StaticArrayInterface.jl. This was separated from ArrayInterface.jl because it includes a lot of functionality that does not give substantive improvements to the interface, and is likely to be deprecated in the near future as the compiler matures to automate a lot of its optimizations.","category":"page"},{"location":"#API","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"API","text":"","category":"section"},{"location":"#Traits","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"Traits","text":"","category":"section"},{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"ArrayInterface.can_avx\nArrayInterface.can_change_size\nArrayInterface.can_setindex\nArrayInterface.device\nArrayInterface.defines_strides\nArrayInterface.ensures_all_unique\nArrayInterface.ensures_sorted\nArrayInterface.fast_matrix_colors\nArrayInterface.fast_scalar_indexing\nArrayInterface.indices_do_not_alias\nArrayInterface.instances_do_not_alias\nArrayInterface.is_forwarding_wrapper\nArrayInterface.ismutable\nArrayInterface.isstructured\nArrayInterface.has_sparsestruct\nArrayInterface.ndims_index\nArrayInterface.ndims_shape\n","category":"page"},{"location":"#ArrayInterface.can_avx","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.can_avx","text":"can_avx(f) -> Bool\n\nReturns true if the function f is guaranteed to be compatible with LoopVectorization.@avx for supported element and array types. While a return value of false does not indicate the function isn't supported, this allows a library to conservatively apply @avx only when it is known to be safe to do so.\n\nfunction mymap!(f, y, args...)\n    if can_avx(f)\n        @avx @. y = f(args...)\n    else\n        @. y = f(args...)\n    end\nend\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.can_change_size","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.can_change_size","text":"can_change_size(::Type{T}) -> Bool\n\nReturns true if the Base.size of T can change, in which case operations such as pop! and popfirst! are available for collections of type T.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.can_setindex","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.can_setindex","text":"can_setindex(::Type{T}) -> Bool\n\nQuery whether a type can use setindex!.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.device","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.device","text":"device(::Type{T}) -> AbstractDevice\n\nIndicates the most efficient way to access elements from the collection in low-level code. For GPUArrays, will return ArrayInterface.GPU(). For AbstractArray supporting a pointer method, returns ArrayInterface.CPUPointer(). For other AbstractArrays and Tuples, returns ArrayInterface.CPUIndex(). Otherwise, returns nothing.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.defines_strides","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.defines_strides","text":"defines_strides(::Type{T}) -> Bool\n\nIs strides(::T) defined? It is assumed that types returning true also return a valid pointer on pointer(::T).\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.ensures_all_unique","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.ensures_all_unique","text":"ensures_all_unique(T::Type) -> Bool\n\nReturns true if all instances of type T are composed of a unique set of elements. This does not require that T subtypes AbstractSet or implements the AbstractSet interface.\n\nExamples\n\njulia> ArrayInterface.ensures_all_unique(BitSet())\ntrue\n\njulia> ArrayInterface.ensures_all_unique([])\nfalse\n\njulia> ArrayInterface.ensures_all_unique(typeof(1:10))\ntrue\n\njulia> ArrayInterface.ensures_all_unique(LinRange(1, 1, 10))\nfalse\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.ensures_sorted","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.ensures_sorted","text":"ensures_sorted(T::Type) -> Bool\n\nReturns true if all instances of T are sorted.\n\nExamples\n\njulia> ArrayInterface.ensures_sorted(BitSet())\ntrue\n\njulia> ArrayInterface.ensures_sorted([])\nfalse\n\njulia> ArrayInterface.ensures_sorted(1:10)\ntrue\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.fast_matrix_colors","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.fast_matrix_colors","text":"fast_matrix_colors(A)\n\nQuery whether a matrix has a fast algorithm for getting the structural colors of the matrix.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.fast_scalar_indexing","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.fast_scalar_indexing","text":"fast_scalar_indexing(::Type{T}) -> Bool\n\nQuery whether an array type has fast scalar indexing.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.indices_do_not_alias","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.indices_do_not_alias","text":"indices_do_not_alias(::Type{T<:AbstractArray}) -> Bool\n\nIs it safe to ivdep arrays of type T? That is, would it be safe to write to an array of type T in parallel? Examples where this is not true are BitArrays or view(rand(6), [1,2,3,1,2,3]). That is, it is not safe whenever different indices may alias the same memory.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.instances_do_not_alias","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.instances_do_not_alias","text":"instances_do_not_alias(::Type{T}) -> Bool\n\nIs it safe to ivdep arrays containing elements of type T? That is, would it be safe to write to an array full of T in parallel? This is not true for mutable structs in general, where editing one index could edit other indices. That is, it is not safe when different instances may alias the same memory.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.is_forwarding_wrapper","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.is_forwarding_wrapper","text":"is_forwarding_wrapper(::Type{T}) -> Bool\n\nReturns true if the type T wraps another data type and does not alter any of its standard interface. For example, if T were an array then its size, indices, and elements would all be equivalent to its wrapped data.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.ismutable","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.ismutable","text":"ismutable(::Type{T}) -> Bool\n\nQuery whether instances of type T are mutable or not, see https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.isstructured","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.isstructured","text":"isstructured(::Type{T}) -> Bool\n\nQuery whether a type is a representation of a structured matrix.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.has_sparsestruct","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.has_sparsestruct","text":"has_sparsestruct(x::AbstractArray) -> Bool\n\nDetermine whether findstructralnz accepts the parameter x.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.ndims_index","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.ndims_index","text":"ndims_index(::Type{I}) -> Int\n\nReturns the number of dimensions that an instance of I indexes into. If this method is not explicitly defined, then 1 is returned.\n\nSee also ndims_shape\n\nExamples\n\njulia> ArrayInterface.ndims_index(Int)\n1\n\njulia> ArrayInterface.ndims_index(CartesianIndex(1, 2, 3))\n3\n\njulia> ArrayInterface.ndims_index([CartesianIndex(1, 2), CartesianIndex(1, 3)])\n2\n\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.ndims_shape","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.ndims_shape","text":"ndims_shape(::Type{I}) -> Union{Int,Tuple{Vararg{Int}}}\n\nReturns the number of dimension that are represented in the shape of the returned array when indexing with an instance of I.\n\nSee also ndims_index\n\nExamples\n\n```julia julia> ArrayInterface.ndims_shape([CartesianIndex(1, 1), CartesianIndex(1, 2)]) 1\n\njulia> ndims(CartesianIndices((2,2))[[CartesianIndex(1, 1), CartesianIndex(1, 2)]]) 1\n\n\n\n\n\n","category":"function"},{"location":"#Functions","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"Functions","text":"","category":"section"},{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"ArrayInterface.allowed_getindex\nArrayInterface.allowed_setindex!\nArrayInterface.aos_to_soa\nArrayInterface.buffer\nArrayInterface.findstructralnz\nArrayInterface.flatten_tuples\nArrayInterface.lu_instance\nArrayInterface.map_tuple_type\nArrayInterface.matrix_colors\nArrayInterface.issingular\nArrayInterface.parent_type\nArrayInterface.promote_eltype\nArrayInterface.restructure\nArrayInterface.safevec\nArrayInterface.zeromatrix\nArrayInterface.undefmatrix","category":"page"},{"location":"#ArrayInterface.allowed_getindex","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.allowed_getindex","text":"allowed_getindex(x,i...)\n\nA scalar getindex which is always allowed.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.allowed_setindex!","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.allowed_setindex!","text":"allowed_setindex!(x,v,i...)\n\nA scalar setindex! which is always allowed.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.aos_to_soa","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.aos_to_soa","text":"aos_to_soa(x)\n\nConverts an array of structs formulation to a struct of array.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.buffer","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.buffer","text":"buffer(x)\n\nReturn the buffer data that x points to. Unlike parent(x::AbstractArray), buffer(x) may not return another array type.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.findstructralnz","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.findstructralnz","text":"findstructralnz(x::AbstractArray)\n\nReturn: (I,J) #indexable objects Find sparsity pattern of special matrices, the same as the first two elements of findnz(::SparseMatrixCSC).\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.flatten_tuples","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.flatten_tuples","text":"ArrayInterface.flatten_tuples(t::Tuple) -> Tuple\n\nFlattens any field of t that is a tuple. Only direct fields of t may be flattened.\n\nExamples\n\njulia> ArrayInterface.flatten_tuples((1, ()))\n(1,)\n\njulia> ArrayInterface.flatten_tuples((1, (2, 3)))\n(1, 2, 3)\n\njulia> ArrayInterface.flatten_tuples((1, (2, (3,))))\n(1, 2, (3,))\n\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.lu_instance","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.lu_instance","text":"luinstance(A) -> lufactorization_instance\n\nReturns an instance of the LU factorization object with the correct type cheaply.\n\n\n\n\n\nlu_instance(a::Number) -> a\n\nReturns the number.\n\n\n\n\n\nlu_instance(a::Any) -> lu(a, check=false)\n\nReturns the number.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.map_tuple_type","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.map_tuple_type","text":"ArrayInterface.map_tuple_type(f, T::Type{<:Tuple})\n\nReturns tuple where each field corresponds to the field type of T modified by the function f.\n\nExamples\n\njulia> ArrayInterface.map_tuple_type(sqrt, Tuple{1,4,16})\n(1.0, 2.0, 4.0)\n\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.matrix_colors","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.matrix_colors","text":"matrix_colors(A::Union{Array,UpperTriangular,LowerTriangular})\n\nThe color vector for dense matrix and triangular matrix is simply [1,2,3,..., Base.size(A,2)].\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.issingular","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.issingular","text":"issingular(A::AbstractMatrix) -> Bool\n\nDetermine whether a given abstract matrix is singular.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.parent_type","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.parent_type","text":"parent_type(::Type{T}) -> Type\n\nReturns the parent array that type T wraps.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.promote_eltype","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.promote_eltype","text":"promote_eltype(::Type{<:AbstractArray{T,N}}, ::Type{T2})\n\nComputes the type of the AbstractArray that results from the element type changing to promote_type(T,T2).\n\nNote that no generic fallback is given.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.restructure","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.restructure","text":"restructure(x,y)\n\nRestructures the object y into a shape of x, keeping its values intact. For simple objects like an Array, this simply amounts to a reshape. However, for more complex objects such as an ArrayPartition, not all of the structural information is adequately contained in the type for standard tools to work. In these cases, restructure gives a way to convert for example an Array into a matching ArrayPartition.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.safevec","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.safevec","text":"safevec(v)\n\nIt is a form of vec which is safe for all values in vector spaces, i.e., if it is already a vector, like an AbstractVector or Number, it will return said AbstractVector or Number.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.zeromatrix","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.zeromatrix","text":"zeromatrix(u::AbstractVector)\n\nCreates the zero'd matrix version of u. Note that this is unique because similar(u,length(u),length(u)) returns a mutable type, so it is not type-matching, while fill(zero(eltype(u)),length(u),length(u)) doesn't match the array type, i.e., you'll get a CPU array from a GPU array. The generic fallback is u .* u' .* false, which works on a surprising number of types, but can be broken with weird (recursive) broadcast overloads. For higher-order tensors, this returns the matrix linear operator type which acts on the vec of the array.\n\n\n\n\n\n","category":"function"},{"location":"#ArrayInterface.undefmatrix","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.undefmatrix","text":"undefmatrix(u::AbstractVector)\n\nCreates the matrix version of u with possibly undefined values. Note that this is unique because similar(u,length(u),length(u)) returns a mutable type, so it is not type-matching, while fill(zero(eltype(u)),length(u),length(u)) doesn't match the array type, i.e., you'll get a CPU array from a GPU array. The generic fallback is u .* u', which works on a surprising number of types, but can be broken with weird (recursive) broadcast overloads. For higher-order tensors, this returns the matrix linear operator type which acts on the vec of the array.\n\n\n\n\n\n","category":"function"},{"location":"#Types","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"Types","text":"","category":"section"},{"location":"","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","text":"ArrayInterface.ArrayIndex\nArrayInterface.GetIndex\nArrayInterface.SetIndex!","category":"page"},{"location":"#ArrayInterface.ArrayIndex","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.ArrayIndex","text":"ArrayIndex{N}\n\nSubtypes of ArrayIndex represent series of transformations for a provided index to some buffer which is typically accomplished with square brackets (e.g., buffer[index[inds...]]). The only behavior that is required of a subtype of ArrayIndex is the ability to transform individual index elements (i.e. not collections). This does not guarantee bounds checking or the ability to iterate (although additional functionality may be provided for specific types).\n\n\n\n\n\n","category":"type"},{"location":"#ArrayInterface.GetIndex","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.GetIndex","text":"GetIndex(buffer) = GetIndex{true}(buffer)\nGetIndex{check}(buffer) -> g\n\nWraps an indexable buffer in a function type that is indexed when called, so that g(inds..) is equivalent to buffer[inds...]. If check is false, then all indexing arguments are considered in-bounds. The default value for check is true, requiring bounds checking for each index.\n\nSee also SetIndex!\n\n!!! Warning     Passing false as check may result in incorrect results/crashes/corruption for     out-of-bounds indices, similar to inappropriate use of @inbounds. The user is     responsible for ensuring this is correctly used.\n\nExamples\n\njulia> ArrayInterface.GetIndex(1:10)(3)\n3\n\njulia> ArrayInterface.GetIndex{false}(1:10)(11)  # shouldn't be in-bounds\n11\n\n\n\n\n\n\n","category":"type"},{"location":"#ArrayInterface.SetIndex!","page":"ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming","title":"ArrayInterface.SetIndex!","text":"SetIndex!(buffer) = SetIndex!{true}(buffer)\nSetIndex!{check}(buffer) -> g\n\nWraps an indexable buffer in a function type that sets a value at an index when called, so that g(val, inds..) is equivalent to setindex!(buffer, val, inds...). If check is false, then all indexing arguments are considered in-bounds. The default value for check is true, requiring bounds checking for each index.\n\nSee also GetIndex\n\n!!! Warning     Passing false as check may result in incorrect results/crashes/corruption for     out-of-bounds indices, similar to inappropriate use of @inbounds. The user is     responsible for ensuring this is correctly used.\n\nExamples\n\n\njulia> x = [1, 2, 3, 4];\n\njulia> ArrayInterface.SetIndex!(x)(10, 2);\n\njulia> x[2]\n10\n\n\n\n\n\n\n","category":"type"}]
}
