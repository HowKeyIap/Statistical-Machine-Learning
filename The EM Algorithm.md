

# The EM Algorithm

##  Description

> ​		The ***Expectation Maximization Algorithm(EM algorithm)***, is a general technique for finding maximum likelihood solutions for probabilistic models having latent variables.
> $$
> \color{red} {\boldsymbol { \theta } ^{(t+1)} = \underset { \theta } { \arg \max } \int_ {\mathbf{Z}} \left[ ( \ln p ( \mathbf { X }, \mathbf { Z } | \boldsymbol { \theta } ) \  * \ p( \mathbf { Z }| \mathbf { X } , \boldsymbol { \theta ^{(t)}})) \right] d\mathbf { Z }}
> $$
> 

|                          Variables                           |                           Meaning                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                        $\mathbf {X}$                         |                      Observed variables                      |
|                        $\mathbf {Z}$                         |                       Hidden variables                       |
|                    $\boldsymbol{\theta}$                     |                          Parameters                          |
|       $p(\mathbf{X}, \mathbf{Z}| \boldsymbol{\theta})$       | Joint distribution governed by parameters $ \boldsymbol{\theta} $ |
| $p ( \mathbf { Z }|\mathbf { X }, \boldsymbol { \theta }  )$ | Conditional distribution of Z given X and $ \boldsymbol{\theta} $ |
|             $p(\mathbf{X}|\boldsymbol{\theta})$              |                     Likelihood function                      |

## Facts

1. Fact 1: [*Jensen's* *inequality*](<https://en.wikipedia.org/wiki/Jensen's_inequality>)

   ​	Let $\forall \ \varphi \in$ be a convex function,  and let $X$ be a random variable. Then:

$$
\varphi (E(x)) \le E(\varphi(x))
$$

​			Notice that equality holds if and only if $X = E[X]$ with probability 1(i.e. if $X$ is constant).

2. Fact 2: Sum and product rules

$$
p ( \mathbf { X } | \boldsymbol { \theta } ) = \int _ { \mathbf { Z } } p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) \ d{ \mathbf { Z } }
$$



3. Fact 3: [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes'_theorem)

$$
\begin{align}  p ( \mathbf { X } | \boldsymbol { \theta } ) & = \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) }  { p ( \mathbf { Z } | \mathbf { X },\boldsymbol { \theta } ) } \\

\ln (p ( \mathbf { X } | \boldsymbol { \theta } ) ) & =  \ln  \left( \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) }  { p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta } ) } \right)  \\

& = \ln \left( \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right) \ - \  \ln \left( \frac { p ( \mathbf { Z } | \mathbf { X }, \boldsymbol { \theta }  ) } { q ( \mathbf { Z } ) } \right) \\
\end{align}
$$

## EM formula derivation

Next we introduce a normalized distribution $q ( \mathbf { Z } )$ that sums to 1 to Equations (3), defined over the latent variables:
$$
\int _ { \mathbf { Z } } { q ( \mathbf { Z } ) } \ d \mathbf { Z }  = 1
$$
We have
$$
\begin{align}
\ln p ( \mathbf { X } | \boldsymbol { \theta } ) & = \ln \int_ \mathbf { Z } p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) \ d\mathbf { Z } \\

& = \ln \int_ \mathbf { Z } \frac {p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } )} {q(\mathbf { Z })} * q(\mathbf { Z }) \ d\mathbf { Z } \\

& = \underbrace{\ln \left\{  \mathbf { E }_{q(\mathbf { Z })} [{\frac {p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } )} {q(\mathbf { Z })} }] \right\} }_{\Downarrow \color{blue}{\color{green}{\ln(E(x)) \ge E[\ln(x)]},\ Jensen’s inequality}}  \\

& \ge  \underbrace{\mathbf { E }_{\mathbf { q(\mathbf { Z }) }} \left\{  \ln  {\frac {p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } )} {q(\mathbf { Z })} }   \right\} }_{\color{green}{ELBO}}
\end{align}
$$
Notice that equality holds if and only if $\frac {p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } )} {q(\mathbf { Z })} = C$, $C$ is constant.
$$
\begin{align}
\frac {p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } )} {q(\mathbf { Z })} & = C \\

q(\mathbf { Z }) & = \frac 1 C * p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) \\

\int_ \mathbf { Z } q(\mathbf { Z })\  d\mathbf { Z } &= \frac 1 C * \int_\mathbf { Z } p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) d\mathbf { Z } \\

1 & = \frac 1 C  * p ( \mathbf { X } | \boldsymbol { \theta } ) \\

p ( \mathbf { X } | \boldsymbol { \theta } ) & = C = \frac {p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } )} {q(\mathbf { Z })} \\

{q(\mathbf { Z })} &= \frac {p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } )} {p ( \mathbf { X } | \boldsymbol { \theta } )} \\

 \color{red}{{q(\mathbf { Z })}} & \  \color{red}{= p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta } )}

\end{align}
$$
The EM algorithm is a two-stage iterative optimization technique for finding maximum likelihood solutions. 
Suppose that the current value of the parameter vector is ${ \theta }  ^{(t)}$ . 

> Repeat until convergence {

- >  <font color=#FF0000 size=3 >Step 1. E Step </font>: the lower bound $\mathcal { L } ( q , \boldsymbol { \theta }^{（t})$ is maximized with respect to q(Z) while holding ${ \theta }  ^{(t)}$ fixed.

  $$
  q(\mathbf { Z }) = p ( \mathbf { Z } | \mathbf { X },\boldsymbol { \theta }^{(t)} )
  $$

- > <font color=#FF0000 size=3 > Step 2. M Step </font>: the distribution $q(\mathbf { Z }) $ is held fixed and the lower bound $\mathcal { L } ( q , \boldsymbol { \theta }^{（t})$is maximized with respect to θ to give some new value ${ \theta }  ^{(t+1)}$

  $$
  \boldsymbol { \theta } ^{(t+1)} = \underset { \theta } { \arg \max } \int_ {\mathbf{Z}} \left[ ( \ln p ( \mathbf { X }, \mathbf { Z } | \boldsymbol { \theta } ) \  * \ p( \mathbf { Z }, \mathbf { X } | \boldsymbol { \theta ^{(t)}})) \right] d\mathbf { Z }
  $$

  > }

## Proof of Convergence

Instead of perform: 
$$
\boldsymbol { \theta } ^ { \mathrm { MLE } } = \underset { \theta } { \arg \max } ( \mathcal { L } ( \theta ) ) = \underset { \theta } { \arg \max } ( \ln p ( \mathbf { X } | \boldsymbol { \theta } ))
$$

1. The trick is to assume "latent" variable $\mathbf { Z }$ to the model
2. such that we generate a series of $\boldsymbol { \theta } = \{ {\boldsymbol { \theta } ^{(1)},\boldsymbol { \theta } ^{(2)}  }, \cdots, \boldsymbol { \theta } ^{(t)} \}$

For each iteration of the E-M algorithm, we perform:
$$
\boldsymbol { \theta } ^{(t+1)} = \underset { \theta } { \arg \max } \int_ {\mathbf{Z}} \left[ ( \ln p ( \mathbf { X }, \mathbf { Z } | \boldsymbol { \theta } ) \  * \ p( \mathbf { Z }| \mathbf { X } , \boldsymbol { \theta ^{(t)}})) \right] d\mathbf { Z }
$$
However, we must ensure convergence for each $t$:
$$
\color{red} {\ln p ( \mathbf { X } | \boldsymbol { \theta }^{(t + 1)}  ) \ge \ln p ( \mathbf { X } | \boldsymbol { \theta }^{(t)}  )}
$$
<font color=#FF0000 size=4 >Proof  </font>
$$
\begin{align}  
p ( \mathbf { X } | \boldsymbol { \theta } ) & = \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) }  { p ( \mathbf { Z } | \mathbf { X },\boldsymbol { \theta } ) } \\
\\
\ln \left[ p ( \mathbf { X } | \boldsymbol { \theta } )\right] &= \ln \left[ { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) }\right] -  \ln \left[ { p ( \mathbf { Z } | \mathbf { X },\boldsymbol { \theta } ) } \right]
\end{align}
$$
Introduce $p ( \mathbf { Z } | \mathbf { X },\boldsymbol { \theta }^{(t)} ) $ to Equation(16), we get
$$
\begin{align}
left &=  \int _  { \mathbf { Z } } \ln (p ( \mathbf { X } | \boldsymbol { \theta } ) ) \ * \ { p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)}) } \ d{ \mathbf { Z } } \\
& = \ln \left[ p ( \mathbf { X } | \boldsymbol { \theta } ) \right] \ * \ \underbrace{\int _ { \mathbf { Z } } { p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)}) } \ d{ \mathbf { Z } }}_{1} \\
& =  \ln \left[ p ( \mathbf { X } | \boldsymbol { \theta } ) \right] \\

right &= \underbrace{\int _  { \mathbf { Z } } \ln \left[ { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) }\right] * p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)}) \ d{ \mathbf { Z } }}_{Q(\boldsymbol { \theta },\boldsymbol { \theta }  ^{(t)})} \\
& - \underbrace{\int _  { \mathbf { Z } } \ln \left[ { p ( \mathbf { Z } | \mathbf { X }, \boldsymbol { \theta } ) }\right] *  p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)}) \ d{ \mathbf { Z } }}_{H(\boldsymbol { \theta },\boldsymbol { \theta }  ^{(t)})} \\

\boldsymbol { \theta } ^{(t+1)} & =  \underset { \theta } { \arg \max } \int_ {\mathbf{Z}} \left[ ( \ln p ( \mathbf { X }, \mathbf { Z } | \boldsymbol { \theta } ) \  * \ p( \mathbf { Z }, \mathbf { X } | \boldsymbol { \theta ^{(t)}})) \right] d\mathbf { Z } \\ 

&= \underbrace {\underset { \theta } { \arg \max } \int_ {\mathbf{Z}} Q(\boldsymbol { \theta },\boldsymbol { \theta }  ^{(t)}) d\mathbf { Z }}_{\color{red} {Q(\boldsymbol { \theta }^{(t+1)},\boldsymbol { \theta }  ^{(t)}) \ge Q(\boldsymbol { \theta }^{(t)},\boldsymbol { \theta }  ^{(t)})}} \\

H(\boldsymbol { \theta }^{(t+1)},\boldsymbol { \theta }  ^{(t)}) - H(\boldsymbol { \theta }^{(t)},\boldsymbol { \theta }  ^{(t)}) &=  \int _  { \mathbf { Z } }  \left( \ln \left[ { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta }^{(t+1)} ) }\right] -  \ln \left[ { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta }^{(t)} ) }\right] \right) *  p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)}) \ d{ \mathbf { Z } } \\

 &=  \int _  { \mathbf { Z } } \ln \frac{  \left[ { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta }^{(t+1)} ) }\right]} {   \left[ { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta }^{(t)} ) }\right] } *  p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)}) \ d{ \mathbf { Z } } \\

& = \underbrace{ \mathbf{E} _ {p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)})} [\ln \frac{  \left[ { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta }^{(t+1)} ) }\right]} {   \left[ { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta }^{(t)} ) }\right] }]}_{\Downarrow {\color{green}{\mathbf{E}[ln(x)] \le ln \mathbf{E}[x] }}} \\

& \le \ln \underbrace{\int _ { \mathbf { Z } } { p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta }  ^{(t)}) } \ d{ \mathbf { Z } }}_{1} \\

& = 0 \Longrightarrow \color{red} {H(\boldsymbol { \theta }^{(t+1)},\boldsymbol { \theta }  ^{(t)}) \le H(\boldsymbol { \theta }^{(t)},\boldsymbol { \theta }  ^{(t)})}

\end{align}
$$
Thus, 
$$
\underbrace{Q(\boldsymbol { \theta }^{(t+1)},\boldsymbol { \theta }  ^{(t)}) - H(\boldsymbol { \theta }^{(t+1)},\boldsymbol { \theta }  ^{(t)}) \ge  Q(\boldsymbol { \theta }^{(t)},\boldsymbol { \theta }  ^{(t)}) - H(\boldsymbol { \theta }^{(t)},\boldsymbol { \theta }  ^{(t)})}_{\color{red}{\ln p ( \mathbf { X } | \boldsymbol { \theta }^{(t + 1)}  ) \ge \ln p ( \mathbf { X } | \boldsymbol { \theta }^{(t)}  )}}
$$

## Other interpretation of ELBO

In Equation (6), we introduce $q ( \mathbf { Z } )$
$$
\begin{align} 
\int _  { \mathbf { Z } } \ln (p ( \mathbf { X } | \boldsymbol { \theta } ) ) \ * \ { q ( \mathbf { Z } ) } \ d{ \mathbf { Z } }\  & = \ \int _ { \mathbf { Z } }  \left[ \ln \left( \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right) \  + \ \ln \left( \frac { q ( \mathbf { Z } ) } { p ( \mathbf { Z } | \mathbf { X },\boldsymbol { \theta }  ) }  \right) \right] \ * \ { q ( \mathbf { Z } ) } \ d{ \mathbf { Z } } \\


\ln (p ( \mathbf { X } | \boldsymbol { \theta } ) ) \ * \ \int _ { \mathbf { Z } } { q ( \mathbf { Z } ) } \ d{ \mathbf { Z } } & =  \int _ { \mathbf { Z } } \left[ \ln \left( \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right) \ * \ { q ( \mathbf { Z } ) } \right] \ d{ \mathbf { Z } } \ \\
& + \int _ { \mathbf { Z } }  \left[ \ \ln \left( \frac { q ( \mathbf { Z } ) } { p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta } ) }  \right) * \ { q ( \mathbf { Z } ) } \right] \ d{ \mathbf { Z } } \\


\ln (p ( \mathbf { X } | \boldsymbol { \theta } ) )  & = \underbrace {\int _ { \mathbf { Z } } \left[ \ln \left( \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right) \ * \ { q ( \mathbf { Z } ) } \right]\ d{ \mathbf { Z } }}_{ELBO} \ \\ 


&+ \underbrace{\int _ { \mathbf { Z } } \left[ \ \ln \left( \frac { q ( \mathbf { Z } ) } { p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta } ) }  \right) * \ { q ( \mathbf { Z } ) } \right]\ d{ \mathbf { Z } }}_{KL(q ( \mathbf { Z } ) , p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta } ))} \\ 
\end{align}
$$

If we define that $\mathcal { L } ( q , \boldsymbol { \theta } )$ is a functional of the distribution $q ( \mathbf { Z } )$ and the parameters $\boldsymbol { \theta }$, $\mathrm { KL } ( q \| p )$ is the  [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) between $q ( \mathbf { Z } )$ and the posterior distribution $p ( \mathbf { Z} | \mathbf { X } ,\boldsymbol { \theta } )  $.
$$
\begin{align}
\mathcal { L } ( q , \boldsymbol { \theta } ) & = \int _ { \mathbf { Z } } q ( \mathbf { Z } ) \ln \left\{ \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right\}  \ d{ \mathbf { Z } } = \mathbf{E} _ {q ( \mathbf { Z } )} \left[ \ln \left\{ \frac { p ( \mathbf { X } , \mathbf { Z } | \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right\} \right]\\
\mathrm { KL } ( q \| p ) & = - \int _ { \mathbf { Z } } q ( \mathbf { Z } ) \ln \left\{ \frac { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right\} \ d{ \mathbf { Z } } =  \mathbf{E} _ {q ( \mathbf { Z } )} \left[ \ln \left\{ \frac { p ( \mathbf { Z } | \mathbf { X } , \boldsymbol { \theta } ) } { q ( \mathbf { Z } ) } \right\} \right] \\
\end{align}
$$
Equation (9) holds that 
$$
\ln p ( \mathbf { X } | \boldsymbol { \theta } ) = \mathcal { L } ( q , \boldsymbol { \theta } ) + \mathrm { KL } ( q \| p )
$$
Recall that the  [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) satisfies $\mathrm { KL } ( q \| p ) = 0$, with equality if, and only if, $q ( \mathbf { Z } )  = p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta } )$, It therefore follows from (8) that $\mathcal { L } ( q , \boldsymbol { \theta } )  \le  \ln p ( \mathbf { X } | \boldsymbol { \theta } ) $, in other words that $\mathcal { L } ( q , \boldsymbol { \theta } ) $ is a lower bound (`ELBO`) on  $\ln p ( \mathbf { Z } | \mathbf { X } ,\boldsymbol { \theta } )$.

# The EM Algorithm for Gaussian Mixture Model (GMM)



## Finite Mixture Models

### Definition

​		We are given a data set  $\mathbf{X} = \left\{ { \boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_i \cdots, \boldsymbol{x}}_N \right\}$, where $\boldsymbol{x_i}$ is a $D$-dimensional vector measurement. Assume that the points are generated in an IID fashion from probability density function $p(\mathbf{X})$.

​	We further assume that $p(\mathbf{X})$ is defined as a finite mixture model by taking linear combinations of more basic distributions ($K$ components) such as Gaussians：
$$
p (  \mathbf{X} _i  | \boldsymbol{\Theta} ) = \sum _ { k = 1 } ^ { K } \underbrace{\alpha _ { k }}_{p_k (\mathbf{Z}_{ik}=1)} * \underbrace{p _ { k } \left(  \mathbf{X}_i |  \mathbf{Z}_{ik}=1 , \boldsymbol{\theta} _ { k } \right)}_{p _ { k } \left(  \mathbf{X}_i |  \mathbf{Z}_{ik}=1 , \boldsymbol{\theta} _ { k } \right)}
$$
​		where: $i \in [1,N], k \in [1,K]$

+ $p _ { k } \left(  \mathbf{X} |  \mathbf{Z}_{k} , \boldsymbol{\theta} _ { k } \right)$ are mixture components. In general, the components can be any distribution or density function, and need not all have the same functional form.

+ $\mathbf{Z}_{i} = \left\{ \mathbf{Z}_{i1},  \mathbf{Z}_{i2},  \cdots, \mathbf{Z}_{ik}, \cdots, \mathbf{Z}_{iK} \right\}  $ plays the role of indicator random variable representing the identity of the mixture component that generated $X_i$. (i.e., one and only one of the $\mathbf{Z}_{i}$ is equal to 1, and the others are 0).

+ $\alpha _k = p(\mathbf{Z}_{ik} = 1)$ are the mixture weights, representing the probability that a randomly selected $i$ was generated by component $k$, where $\sum_{k=1}^{K} \alpha_k = 1$
+ $\mathbf{\Theta} = \left\{ \alpha_1, \cdots, \alpha_K, \boldsymbol{\theta}_1, \cdots,  \boldsymbol{\theta}_K \right\}$  is a the complete set of a parameters for a mixture model with $K$ components.

### Application

+ One general application is in ***density estimation***: they allow us to build complex models out of simple parts. For example, a mixture of K multivariate Gaussians may have up to $K$ modes, allowing us to model multimodal densities.
+  A second motivation for using mixture models is where there is an ***underlying true categorical variable z***, but we cannot directly measure it: a well-known example is pulling fish from a lake and measuring their weight $\mathbf{X}_i$ ,  where there are known to be K types of fish in the lake but the types were not measured in the data we were given. In this situation the $\mathbf{Z} _i$ correspond to some actual physical quantity that could have been measured but that wasn't.
+ A third motivation, similar to the second, is where we believe their might be K underlying groups in the data, each characterized by different parameters, e.g., K sets of customers which we wish to infer from purchasing data xi. This is often referred to as ***model-based clustering***: there is not necessarily any true underlying interpretation to the z’s, so this tends to be more exploratory in nature than in the second case.

## Gaussian Mixture Models

​		For $\boldsymbol{x}_i \in \mathcal{R}^D$, we definite a Gaussian mixture model by making each of the $K$ components a Gaussian density with parameters $\mu _k$ and $\Sigma _k$. Each component is a multivariate Gaussian density with its own parameters $\boldsymbol{\theta}_k = \left\{  \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k \right\}$.
$$
\mathcal { N } (\boldsymbol{x}_i | \boldsymbol{\theta}_k ) = \frac { 1 } { ( 2 \pi ) ^ { D / 2 } } \frac { 1 } { | \boldsymbol { \Sigma }_k | ^ { 1 / 2 } } \exp \left\{ - \frac { 1 } { 2 } ( \boldsymbol{x}_i - \boldsymbol { \mu }_k ) ^ { \mathrm { T } } \boldsymbol { \Sigma }_k ^ { - 1 } ( \boldsymbol{x}_i - \boldsymbol { \mu }_k ) \right\}
$$
​	where $\boldsymbol { \mu }$ is a $D$-dimensional mean vector, $\boldsymbol{\Sigma}$ is a $D \times D$ covariance matrix, and $|\boldsymbol{\Sigma}|$
denotes the determinant of $\boldsymbol{\Sigma}$ . 

​	 We therefore consider the superposition of  $K$ Gaussian densities of the form
$$
\begin{align} p_k ( \boldsymbol { x }_i |\boldsymbol{\theta}) &= \sum _ { k = 1 } ^ { K } \boldsymbol{\pi} _ { k } \mathcal { N } \left( \boldsymbol { x }_i | \boldsymbol { \mu } _ { k } , \mathbf { \Sigma } _ { k } \right)\\

 p ( \mathbf { X } |\boldsymbol{\theta}) &= \prod _ { i = 1 } ^ { N }   \sum _ { k = 1 } ^ { K } \left[ \boldsymbol{\pi} _ { k } \mathcal { N } \left( \boldsymbol { x }_i | \boldsymbol { \mu } _ { k } , \mathbf { \Sigma } _ { k } \right) \right]
\end{align}
$$
​		where, $\mathbf{X} = \left\{ { \boldsymbol{x}_1,  \ldots, \boldsymbol{x}}_N \right\}, \boldsymbol { \pi } = \left\{ \boldsymbol{\pi} _ { 1 } , \ldots , \boldsymbol{\pi} _ { K } \right\} , \boldsymbol { \mu } = \left\{ \boldsymbol { \mu } _ { 1 } , \ldots , \boldsymbol { \mu } _ { K } \right\}$ and $
\boldsymbol{\Sigma} \equiv \left\{\boldsymbol{ \Sigma} _ { 1 } , \dots \boldsymbol{\Sigma} _ { K } \right\}$, $\boldsymbol{\theta} = \left\{ \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma} \right\} $. $i \in [1,N], k \in [1,K]$.  $\sum_{k=1}^{K} \boldsymbol{\pi}_k = 1$

​		We immediately see that the situation is now much more complex than with a single Gaussian, due to the presence of the summation over k inside the logarithm. As a result, <u>the maximum likelihood solution for the parameters no longer has a closed-form analytical solution</u>.



### The E-Step for Mixture Models

we introduce “latent” variable $\mathbf{Z} = \left\{ { \boldsymbol{z}_1,  \ldots, \boldsymbol{z}}_N \right\}, $ each $\boldsymbol{z}_i$ indicates which mixture component $\boldsymbol{x}_i$ belong to. Recall the EM algorithm in Equation (1)， we need to define both $p ( \mathbf { X }, \mathbf { Z } | \boldsymbol { \theta } ) \  $ and $ \ p( \mathbf { Z }| \mathbf { X } , \boldsymbol { \theta ^{(t)}})$,
$$
\begin{align}

p _k \left(  \mathbf{Z}_i | \boldsymbol{\theta} \right) & = \boldsymbol{\pi} _ { k } \\

p_k \left( \mathbf{X}_i | \boldsymbol{\theta} \right) &= \sum _{i=1} ^{N} p _k \left( \mathbf{X}_i, \mathbf{Z}_i|\boldsymbol{\theta} \right)\\

& = \sum _{i=1} ^{K} p _k \left( \mathbf{X}_i| \mathbf{Z}_i, \boldsymbol{\theta}\right) p _k \left(  \mathbf{Z}_i | \boldsymbol{\theta} \right)\\

p \left( \mathbf{X}_i| \mathbf{Z}_i, \boldsymbol{\theta} \right) &= \sum _{i=1} ^{N} p _k \left( \mathbf{X}_i| \mathbf{Z}_i, \boldsymbol{\theta} \right)\\
p(x) = \sum_z p(x,z) = \sum p(x|z)p(z)\\

p \left( \mathbf{X}_i, \mathbf{Z}_i|\boldsymbol{\theta} \right) &= \sum _{i=1} ^{K} p _k \left( \mathbf{X}_i, \mathbf{Z}_i|\boldsymbol{\theta} \right)\\

& = \sum _{i=1} ^{K} p _k \left( \mathbf{X}_i| \mathbf{Z}_i, \boldsymbol{\theta}\right) p _k \left(  \mathbf{Z}_i | \boldsymbol{\theta} \right)\\

p \left( \mathbf{X}, \mathbf{Z}| \boldsymbol{\theta}\right) &= \prod _{i=1} ^{N} p \left( \mathbf{X}_i, \mathbf{Z}_i| \boldsymbol{\theta}\right)\\

& = \prod _{i=1} ^{N} \left\{ \sum _{i=1} ^{K}  p _k \left( \mathbf{Z}_i| \mathbf{X}_i, \boldsymbol{\theta} \right)  p _k \left( \mathbf{Z}_i| \mathbf{X}_i, \boldsymbol{\theta} \right) \right\}\\

&= \prod _ { i = 1 } ^ { N }   \sum _ { k = 1 } ^ { K }  \mathcal { N } \left( \boldsymbol { x }_i | \boldsymbol { \mu } _ { k } , \mathbf { \Sigma } _ { k } \right) \boldsymbol{\pi} _ { k }\\

p \left( \mathbf{Z}| \mathbf{X}, \boldsymbol{\theta}\right) & = \prod _{i=1} ^{N}  p \left( \mathbf{Z}_i| \mathbf{X}_i, \boldsymbol{\theta}\right)\\

&= \prod _{i=1} ^{N} \frac   {p \left( \mathbf{X}_i, \mathbf{Z}_i, \boldsymbol{\theta} \right)}  {p \left( \mathbf{X}_i, \boldsymbol{\theta} \right)}   \\

& = \prod _{i=1} ^{N} \frac   {\sum _{i=1} ^{K} p _k \left( \mathbf{X}_i, \mathbf{Z}_i, \boldsymbol{\theta} \right)}  {\sum _{i=1} ^{K} p_k \left( \mathbf{X}_i, \mathbf{Z}_i|\boldsymbol{\theta} \right)}   \\

&= \prod _ { i = 1 } ^ { N }  \frac {\mathcal { N } \left( \boldsymbol { x }_i | \boldsymbol { \mu } _ { k } , \mathbf { \Sigma } _ { k } \right) \boldsymbol{\pi} _ { k }} {\sum _ { k = 1 } ^ { K }  \mathcal { N } \left( \boldsymbol { x }_i | \boldsymbol { \mu } _ { k } , \mathbf { \Sigma } _ { k } \right) \boldsymbol{\pi} _ { k }}\\



\end{align}
$$
​				


​					
​				
​				
​						
​				
​			


​				
​				


​				
​						
​				
​			


​				
​				



