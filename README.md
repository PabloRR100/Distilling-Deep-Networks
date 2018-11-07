# Distilling-Deep-Networks

The idea of this project is to create a python decorator that could be plugged into a model (should be PyTorch with specific configuration to define the layers) and will inherit the model adding the necessary attributes and methods to keep track of the training statistics and build an interactive dashboard to explore those stats.  

Previous to the construction of the dashboard it is essential to determine the statistics we are interested in:  
- Distribution of weights  
- Distribution of the gradients  
- Evolution of the ratio of weights and update steps   
- Code and try different Optimizers to compare statistics in different regimes    


---


Analyze how different combinations of architectures of networks, activation functions, learning_rate, batch_size change the perfomarnce on the test inference evaluating next picture, and the evolution of the gradients and the activations to see how the flow during the forward and backward passes respectively.

The examples shown will be for a simple 1 hidden layer CNN (Note 1 hidden layer has 2 layers of parameters W1 and W2)  

![test][test_inference]

Example of analyzing the weights:
![weights][weight_stats]

Example of analyzing the gradients:
![grads][grads]
![grads2][grads2]

Example of analyzing the activations:
![activations][activations]

[test_inference]: https://github.com/PabloRR100/Distilling-Deep-Networks/blob/master/figures/pred_results.png?raw=true  
[weight_stats]: https://github.com/PabloRR100/Distilling-Deep-Networks/blob/master/figures/1_weight_stats.png?raw=true  
[grads]: https://github.com/PabloRR100/Distilling-Deep-Networks/blob/master/figures/2_grads_stats_2.png?raw=true
[grads2]: https://github.com/PabloRR100/Distilling-Deep-Networks/blob/master/figures/4_grad_stats.png?raw=true
[activations]: https://github.com/PabloRR100/Distilling-Deep-Networks/blob/master/figures/3_activation_stats.png?raw=true
