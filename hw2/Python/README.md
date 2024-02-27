## README - Object Recognition Criteria 

### 1. Combination Criteria

For the criteria, I employ the relative difference in roundness between two objects, defined as follows:
$$
\Delta_R = \frac{|r_1 - r_2|}{\frac{1}{2}(r_1 + r_2)}
$$
Here, $r_1,r_2$ are the roundness measures of the two objects. The subscript $R$ means "relative". Utilizing absolute difference instead of relative difference fails to distinguish between spoons, forks, and knives in the images.

### 2. Threshold

I selected $0.1$ as the threshold, and it proved to be effective in this practice.