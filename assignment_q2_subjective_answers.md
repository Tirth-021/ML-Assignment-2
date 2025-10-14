## Output:

- In given 200000 iterations we didn't reach convergence criteria
- Iterations required (Unscaled): 200000
- Iterations required (Scaled): 65

$\theta^*$ (Unscaled):
 [[2.21509616]
 [2.99954023]]

$\theta$ (Unscaled):
 [[-0.16075589]
 [ 3.00315907]]

$\theta^*$ (Scaled):
 [[1412.54114977]
 [ 887.85858784]]

$\theta$ (Scaled):
 [[1412.54044108]
 [ 887.85814168]]

---
---
<br>

**Unscaled Data**
- Since my feature values are large, my gradient steps are like taking giant leaps — so we reduced the learning rate to make smaller, safer steps toward the minimum
- But by doing that convergence will be slower, tried running without maximum iteration criteria for more than 1 hour in colab but still didn't converge.

**Scaled data**
- After scaling 99.7% data will lie in [-3, 3] so no problem of divergence while taking 0.01 learning rate. 
- It converged in 65 iterations.