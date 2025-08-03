import cupy as cp
import numpy as np
def vector_add_gpu(a, b):
    """
    Add two arrays element-wise using a CuPy kernel.
    
    Args:
        a, b: Input arrays (numpy or cupy arrays)
    
    Returns:
        cupy array with element-wise sum
    """
    # Convert to CuPy arrays if needed
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    
    # Define the kernel
    add_kernel = cp.ElementwiseKernel(
        'float32 a, float32 b',  # input types
        'float32 c',             # output type
        'c = a + b',             # operation
        'vector_add'             # kernel name
    )
    
    # Execute and return result
    return add_kernel(a_gpu, b_gpu)

# a = np.array(1024, dtype=np.float32)
# vector_add_gpu(a, a)