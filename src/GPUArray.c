#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <julia/julia.h>

typedef struct {
	int64_t n_dim;
	int64_t* dim;
	void** mem;
} jlgpu_array_t;

struct node_t {
	jlgpu_array_t* item;
	struct node_t* next;
};
typedef struct node_t node_t;

typedef struct {
	node_t* first;
} gpumem_t;

node_t __first = {NULL, NULL};
gpumem_t gpumem = {&__first};

void gpumem_append(jlgpu_array_t* item){
	node_t* node = gpumem.first;
	while (node->next != NULL)
		node = node->next;

	node->next = (node_t*) malloc(sizeof(node_t*));
	node->next->next = NULL;
	node->next->item = item;
}

node_t* __gpumem_find(jlgpu_array_t* item){
	node_t* node = gpumem.first;
	while (node->next != NULL)
		if (node->next->item == item)
			return node;

	return NULL;
}

void free_array(jlgpu_array_t* item){
	free(item->mem);
	free(item);
}

void gpumem_delete(jlgpu_array_t* item){
	node_t* node = __gpumem_find(item);
	if (node != NULL) {
		node_t* dnode = node->next;
		node->next = node->next->next;

		free_array(dnode->item);
		free(dnode);
	} else {
		printf("Warning: Couldn't find item.\n");
	}
}

int64_t allocate_array(jl_tuple_t* d){
	jl_(d);
	printf("length: %d\n", jl_tuple_len(d));

	int64_t i;
	for(i=0; i < d->length; ++i)
		printf("%d\n", jl_unbox_int64( jl_tupleref(d,i) ) );

	//TODO: Check tuple values are positive.

	jlgpu_array_t* item = (jlgpu_array_t*) malloc(sizeof(jlgpu_array_t*));
	item->n_dim = jl_tuple_len(d);
	item->dim = (int64_t*) malloc(sizeof(int64_t)*item->n_dim);
	int64_t size = 1;
	for(i=0; i<item->n_dim; ++i){
		item->dim[i] = jl_unbox_int64(jl_tupleref(d,i));
		size *= item->dim[i];
	}

	item->mem = malloc(sizeof(float)*size);

	gpumem_append(item);
	return (int64_t) item;
}



int64_t gemm(jl_array_t* array){
	printf("%f %f\n", ((float*)array->data)[0], ((float*)array->data)[1]);

	cudaError_t cuda_status;
	cublasStatus_t cublas_status;
	cublasHandle_t cublas_handle;

	
	int64_t n = 4000;
	int64_t m = 4000;
	float*	matrix = malloc( sizeof(float) * n*m );


	float* gpu_matrix;

	cuda_status = cudaMalloc((void**)&gpu_matrix, m*n*sizeof(*matrix));

	cublas_status = cublasCreate(&cublas_handle);
	cublas_status = cublasSetMatrix(m, n, sizeof(*matrix), matrix, m, gpu_matrix, m);

	float alpha = 1.0;
	float beta = 1.0;
	cublas_status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &alpha, gpu_matrix, m, gpu_matrix, m, &beta, gpu_matrix, m);

	cublas_status = cublasGetMatrix(m, n, sizeof(*matrix), gpu_matrix, m, matrix, m);

	cudaFree(gpu_matrix);

	return 0;
}

void gpu_free(){
	printf("feeing memory\n");
}