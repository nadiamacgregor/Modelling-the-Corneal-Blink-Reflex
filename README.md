# Modelling the Corneal Blink Reflex Circuitry Using a Neural Network 

**Purpose**: Accurate models of human circuits can provide research tools that allow for neuron level investigation into human anatomy without invasive and expensive real human reserach. 

**Methodology Summary:** 
1) Creation of 4-layer neural network
2) Train neural network on stimuli-response pairs from human patients (eg left eye stimulated, both eyes blink)
3) Keep biases and weights from top three performing neural networks
4) Artifically "lesion" top three performing neural networks with the lowest loss by cutting edges between nodes
5) Test if neural network lesion results in simillar deficits to the corneal blink reflex damage in humans
