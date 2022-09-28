# Transformer
A simple implementation of TRANSFORMER

## `transformer.py`

## `einsum.ipynb` 
This file explains the operation of `torch.einsum()`
> `torch.einsum(equation, *operands) `

- **Free Indices** are the indices specified in the output
- **Summation Indices** are all other indices. Those that appear int the input argument but **NOT** in output specification.


>`M=np.einsum('ij,jk->ij',A,B)`
>- free indices: i,j
>- summation index: k
>```
>A=np.random.rand(3,5)
>B=np.random.rand(5,2)
>M=np.empty((3,2))
>for i in range(3):
>   for j in range(2):
>       total=0
>       for k in range(5):
>           total+=A[i,k]*B[k,j]
>       M[i,j]=total
>```

**RULEs**：
1. Repeating letters in different inputs means those values will be multiplied and those products will be the output.
    `M=np.einsum('ik,kj->ij',A,B)`
2. Omitting a letter means that axis will be summed.
    ```
   X=np.ones(3)
   SUM_X=np.einsum('i->',X)
   ```
3. We can return the unsummed axes in any order.
    ```
   X=np.ones((5,4,3))
   SUM_X=np.einsum('ijk->kji',X)
   ```