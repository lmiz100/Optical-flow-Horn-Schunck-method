# Optical flow: Horn–Schunck method

## Description
Python implementation of [Horn–Schunck method](https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method) for estimating optical flow. <br />

Use **computeHS** to compute u,v optical flow vectors and draw quiver. <br />
`computeHS(name1, name2, alpha, delta)` <br />
Input: images name, smoothing parameter, tolerance <br />
Output: images variations <br />
The paramter 'alpha' is the regularization constant. It determines the smoothness of the output: The bigger
this parameter is, the smoother the solutions we obtain (more locally consistent vectors of motion flow).


## Example
From project directory type in console: `MyHornSchunck.py car1.jpg car2.jpg` <br />
Output: <br />
<p align="center">
  <img width="500" height="500" src="https://github.com/lmiz100/Otical-flow-Horn-Schunck-method/blob/master/results/car%20res.png?raw=true">
</p>

Anothr test options: <br />
`MyHornSchunck.py sphere1.bmp sphere2.bmp`  <br />
`MyHornSchunck.py table1.jpg table2.jpg` <br />
See [results](https://github.com/lmiz100/Otical-flow-Horn-Schunck-method/tree/master/results).
