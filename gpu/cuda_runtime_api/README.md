Experiments on a Domain Specific Language for stencil computation using
expression templates on both CPU and GPU.

## Kernel definition:

```c++
#define STENCIL_EXPRESSION ( (-1.f/12.f) * NW - (1.f/6.f) * N - (1.f/12.f) * NE \
                             -(1.f/6.f)  * W  + CENTER - (1.f/6.f) * E          \
                             -(1.f/12.f) * SW - (1.f/6.f) * S - (1.f/12.f) * SE )
                             
//OR
#define STENCIL_EXPRESSION_SIMPLIFIED ( (-1.f/12.f) * ( NW + NE + SW + SE ) - (1.f/6.f) * ( N + E + W + S ) + CENTER )
```

## Kernel Invocation

```c++
Stencil2D<<<b,tpb>>>( dom, dom2, nrows, ncolumns, STENCIL_EXPRESSION );
```
