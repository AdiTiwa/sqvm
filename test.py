def test_tensor():
    import numpy as np
    from huaq.gates import tensor_product, tensor_prod
    from huaq.utils import algorithm_time

    def tensor1():
        mat1 = np.random.randint(5, size=(128, 128))
        mat2 = np.random.randint(5, size=(2, 2))
        
        tensor = tensor_product(mat1, mat2)

    def tensor2():
        mat1 = np.random.randint(5, size=(128, 128))
        mat2 = np.random.randint(5, size=(2, 2))
        
        tensor = tensor_prod(mat1, mat2)

    print(algorithm_time(tensor1))
    print(algorithm_time(tensor2))
