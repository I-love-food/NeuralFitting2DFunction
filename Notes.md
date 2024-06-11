### Date: 2024.6.10, Monday

**Recent:**
1. Do not use regular lattice as the sample, try using blue noise, poisson disk to generate random samples
2. Try some new powerful libraries, like CGAL (C++), scipy-spatial (python) for specific tasks like triangulation
3. Fix the line trace part, and note that gradient field is a `conservative field` and there is no loops in gradient field. ✔
4. Change the recursive function `trace_on_mesh` to iterative function to find integral line
5. Find a way to calculate the Gradient (Jacobi matrix), Hessian, like this `d(NN) / d(input)`
6. Find a lib to do incremental triangulation
7. Draw the actual integral line path (potential lib: tricontourf, matplotlib can also do a great job) ✔
8. Do not need test set to validate the model

**Near future:**
1. Visualize segmentations on the tri mesh, eg. same color for points with same max label
2. Extend `1.` to min label, min max label
3. Extract critical points: max, min, saddle 
4. Figure out the boundary between segmentations

**Raw notes:**
```python
# sample
# varify triangulation (show the surface in triangulated way)
# integral curve visualize
# visualize segmentation on the tri mesh, trace max, max label
# ... min label
# ... min max label
# boundary between segmentations (boundary is not along the edge possibly)
# integral line (numerial version, trace the gradient, currently, it's not)
# comp. integral line & discrete line (calculate distance, eg. maximal distance, hausdorff distance, Fréchet distance)
# labels on vertices of tri: consistant, inconsistant (refine)
# piecewise linear 可以直接得到
# follow the paper:
# 提取所有的critical point
# quasi critical point
```