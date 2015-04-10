type GPUArray{T<:FloatingPoint)
	id = ccall((:new_array, "libGPUArray"), )

	finalizer(gpu, gpu -> ccall((:gpu_free, "libGPUArray"), Void, ()))
end


immutable type GPUArray{T<:FloatingPoint, N}
	id::Int
	dim::NTuple{N, Int}

	GPUArray(d::NTuple{N, Int}) = ( id = ccall((:allocate_array, "libGPUArray"), Int, (Any,) , (&d)); new(id, d) )
end

GPUArray{T<:FloatingPoint, N}(n::Int...) = GPUArray{T,N}(n)