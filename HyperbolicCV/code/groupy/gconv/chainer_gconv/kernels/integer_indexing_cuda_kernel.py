import cupy as cp

# This computes input[..., T, U, V].swapaxes(1, 2)
_index_group_func_str = \
    """
    extern "C" __global__ void indexing_kernel(
        CArray<{0}, 5> input,
        CArray<int, 4> T,
        CArray<int, 4> U,
        CArray<int, 4> V,
        CArray<{0}, 6> output)
    {{
        CUPY_FOR(i, output.size()) {{

            const int* oshape = output.shape();
            const int* ostrides = output.strides();

            // The flat index i corresponds to the following multi-index in the output array:
            // (output_channel, output_transform, input_channel, input_transform, u, v)
            const int output_channel =   (sizeof({0}) * i / ostrides[0]) % oshape[0];
            const int output_transform = (sizeof({0}) * i / ostrides[1]) % oshape[1];
            const int input_channel =    (sizeof({0}) * i / ostrides[2]) % oshape[2];
            const int input_transform =  (sizeof({0}) * i / ostrides[3]) % oshape[3];
            const int u =                (sizeof({0}) * i / ostrides[4]) % oshape[4];
            const int v =                (sizeof({0}) * i / ostrides[5]) % oshape[5];

            int indexTUV[4] = {{output_transform, input_transform, u, v}};
            int index[5] = {{output_channel, input_channel, T[indexTUV], U[indexTUV], V[indexTUV]}};
            output[i] = input[index];
        }}
    }}
    """

# Use RawModule to compile and access the kernel
module = cp.RawModule(code=_index_group_func_str.format('float'))
_index_group_func_kernel32 = module.get_function('indexing_kernel')
module64 = cp.RawModule(code=_index_group_func_str.format('double'))
_index_group_func_kernel64 = module64.get_function('indexing_kernel')


def index_group_func_kernel(input, T, U, V, output):
    if input.dtype == 'float32':
        _index_group_func_kernel32((output.shape[0],), (output.size,), (input, T, U, V, output))
    elif input.dtype == 'float64':
        _index_group_func_kernel64((output.shape[0],), (output.size,), (input, T, U, V, output))
    else:
        raise ValueError("Unsupported data type")


# Grad indexing kernel
_grad_index_group_func_str_double = \
    """
    // atomicAdd for doubles is not implemented in cuda, so have to add it here
    __device__ double my_atomicAdd(double* address, double val)
    {{
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {{
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));

        }} while (assumed != old);

        return __longlong_as_double(old);
    }}

    extern "C" __global__ void grad_indexing_kernel(
        CArray<{0}, 6> grad_output,
        CArray<int, 4> T,
        CArray<int, 4> U,
        CArray<int, 4> V,
        CArray<{0}, 5> grad_input)
    {{
        CUPY_FOR(i, grad_output.size()) {{

            const int* oshape = grad_output.shape();
            const int* ostrides = grad_output.strides();

            // The flat index i corresponds to the following multi-index in the output array:
            // (output_channel, output_transform, input_channel, input_transform, u, v)
            const int output_channel =   (sizeof({0}) * i / ostrides[0]) % oshape[0];
            const int output_transform = (sizeof({0}) * i / ostrides[1]) % oshape[1];
            const int input_channel =    (sizeof({0}) * i / ostrides[2]) % oshape[2];
            const int input_transform =  (sizeof({0}) * i / ostrides[3]) % oshape[3];
            const int u =                (sizeof({0}) * i / ostrides[4]) % oshape[4];
            const int v =                (sizeof({0}) * i / ostrides[5]) % oshape[5];

            int indexTUV[4] = {{output_transform, input_transform, u, v}};
            int index[5] = {{output_channel, input_channel, T[indexTUV], U[indexTUV], V[indexTUV]}};
            my_atomicAdd(&grad_input[index], grad_output[i]);
        }}
    }}
    """

_grad_index_group_func_str_float = \
    """
    extern "C" __global__ void grad_indexing_kernel(
        CArray<{0}, 6> grad_output,
        CArray<int, 4> T,
        CArray<int, 4> U,
        CArray<int, 4> V,
        CArray<{0}, 5> grad_input)
    {{
        CUPY_FOR(i, grad_output.size()) {{

            const int* oshape = grad_output.shape();
            const int* ostrides = grad_output.strides();

            // The flat index i corresponds to the following multi-index in the output array:
            // (output_channel, output_transform, input_channel, input_transform, u, v)
            const int output_channel =   (sizeof({0}) * i / ostrides[0]) % oshape[0];
            const int output_transform = (sizeof({0}) * i / ostrides[1]) % oshape[1];
            const int input_channel =    (sizeof({0}) * i / ostrides[2]) % oshape[2];
            const int input_transform =  (sizeof({0}) * i / ostrides[3]) % oshape[3];
            const int u =                (sizeof({0}) * i / ostrides[4]) % oshape[4];
            const int v =                (sizeof({0}) * i / ostrides[5]) % oshape[5];

            int indexTUV[4] = {{output_transform, input_transform, u, v}};
            int index[5] = {{output_channel, input_channel, T[indexTUV], U[indexTUV], V[indexTUV]}};
            atomicAdd(&grad_input[index], grad_output[i]);
        }}
    }}
    """

# Compile both functions for float and double
_grad_index_group_func_kernel32 = cp.RawModule(code=_grad_index_group_func_str_float.format('float')).get_function('grad_indexing_kernel')
_grad_index_group_func_kernel64 = cp.RawModule(code=_grad_index_group_func_str_double.format('double')).get_function('grad_indexing_kernel')


def grad_index_group_func_kernel(grad_output, T, U, V, grad_input):
    if grad_output.dtype == 'float32':
        _grad_index_group_func_kernel32((grad_output.shape[0],), (grad_output.size,), (grad_output, T, U, V, grad_input))
    elif grad_output.dtype == 'float64':
        _grad_index_group_func_kernel64((grad_output.shape[0],), (grad_output.size,), (grad_output, T, U, V, grad_input))
    else:
        raise ValueError("Unsupported data type")
