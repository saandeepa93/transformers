# **Important points**

## **1. Attention**
  * Input: X -> (b, s, k)
  ```
  X: (b, s, k)
  b: batch 
  s: sequence length
  k: dimension of each row in sequence
  ```

  * We create 3 more vectors which are scaled projections of the input
  ``` 
  K: Wk * X; (b, s, k)
  Q: Wq * X; (b, s, k)
  V: Wv * X; (b, s, k)
  ```

  * The actual self attention is given by
  ```
  attn: softmax((Q * K.T)/sqrt(k)); (b, s, s)   
  ```
  $$
  X = \begin{bmatrix}
  x_{00} & x_{01} & x_{02}\\
  x_{10} & x_{11} & x_{12}
  \end{bmatrix}\\
   Q = \begin{bmatrix}
  q_{00} & q_{01} & q_{02}\\
  q_{10} & q_{11} & q_{12}
  \end{bmatrix}\\
  K = \begin{bmatrix}
  k_{00} & k_{01} & k_{02}\\
  k_{10} & k_{11} & k_{12}
  \end{bmatrix}\\
  V = \begin{bmatrix}
  v_{00} & v_{01} & v_{02}\\
  v_{10} & v_{11} & v_{12}
  \end{bmatrix}\\
  $$

  $$
  attention = \begin{bmatrix}
  Q_{0*} \times K_{0*}.T & Q_{0*} \times K_{1*}.T\\
  Q_{1*} \times K_{0*}.T & Q_{1*} \times K_{1*}.T
  \end{bmatrix} = 
  \begin{bmatrix}
  a_{00} & a_{01}\\
  a_{10} & a_{11}
  \end{bmatrix}$$

  * Create output data using this attention score
  ```
  Z: X * attn; (b, s, k)
  ```
