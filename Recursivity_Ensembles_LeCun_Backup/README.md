# Recursive Networks

Empirical analysis on ["Understanding Deep Architectures using a Recursive Convolutional Network"][paper]   

![recursive][recursive_img]

Get insights of how recursive networks works, under which circusntances outperform original models and how they could serve as a simulation on a increase of the network budget (number of parameters).   

Example of 3 Networks on a Binary Classification.
 - Original Network: Number Hidden Layers = 2, Hidden Layers Width = 4
 - Recursive Networks: Extend last layer by 2 and 5 recursions

![analysis][recursiveanalysis]

[recursive_img]: https://github.com/PabloRR100/Recursive_Networks/blob/master/figures/recursive.png?raw=true
[recursiveanalysis]: https://github.com/PabloRR100/Recursive_Networks/blob/master/figures/recursive_h2_w4.png?raw=true  
[paper]: https://arxiv.org/abs/1312.1847
