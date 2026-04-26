import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Corneal_Data_Creation import x_lesion, y_lesion, y_L_sensory, x_L_sensory, y_R_sensory, x_R_sensory, y_L_muscle, x_L_muscle, y_R_muscle, x_R_muscle

#activation functions
def relu (x):
    return np.maximum(0,x)
def sigmoid(x):
        return 1/(1+np.exp(-x))
#Parameters
    
i_nodes = 2
h1_nodes = 2
h2_nodes = 2
o_nodes = 2
#define neural net object 

class Net:
    def __init__ (self, w1, w2, w3, b1, b2, b3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
    
        
    def forward (self, inputs, outputs):
        #forward pass
        h1 = relu(inputs.dot(self.w1) + self.b1)
        h2 = relu(h1.dot(self.w2) + self.b2)
        o_preds = sigmoid(h2.dot(self.w3) + self.b3)
        
        #binary cross entropy loss
        loss = -np.mean(outputs * np.log(o_preds + 1e-8) + (1 - outputs) * np.log(1 - o_preds + 1e-8)) 
        
        #convert binary predictions to 0 or 1 
        pred_binary = (o_preds > 0.5).astype(int)
        
        #compute accuacy 
        correct = np.sum(np.all(pred_binary == outputs, axis=1))
        total = len(outputs)
        accuracy = np.mean(pred_binary == outputs) * 100
        return loss, accuracy, o_preds, pred_binary 
        
    def graph(self):
        vis_net([self.w1, self.w2, self.w3], [self.b1, self.b2, self.b3])
    
        
#lesions of network
    def lesion(self, lesion_type):
        if lesion_type == "L_sensory":
                for j in range(2):
                    self.w1[j][0] = 0
        
        elif lesion_type == "R_sensory":
                for i in range(2):
                    self.w1[i][1] = 0 
        
        
        elif lesion_type == "L_trigeminal":
                for n in range(2):
                    self.w2[n][0] = 0 
                
        
        elif lesion_type == "R_trigeminal":
                for g in range(2):
                    self.w2[g][1] = 0 
        
        elif lesion_type == "L_motor":
                for r in range(2):
                    self.w3[r][0] = 0 
        

        elif lesion_type == "R_motor":
                for q in range(2):
                    self.w3[q][1] = 0 
    
        elif lesion_type == "L_muscle":
                for w in range(2):
                    self.w3[0][w] = 0
    
        elif lesion_type == "R_muscle":
                for x  in range(2):
                    self.w3[1][x] = 0 
 

#visulization function 

def vis_net (weights, biases=None, node_radius=300):
    """
    Visualize a feedforward neural network using only weights (and optionally biases).
    - Green edges = positive weights
    - Red edges   = negative weights
    - Edge thickness = proportional to |weight|
    - Zero-weight edges are not drawn
    - Nodes all have the same neutral color
    """

    # --- Validation ---
    if biases is not None and len(weights) != len(biases):
        raise ValueError("Number of weight matrices must equal number of bias vectors.")
    
    # --- Detect orientation (auto-fix transposed weights) ---
    layer_sizes_forward = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    if biases is not None:
        bias_lengths = [len(b) for b in biases]
        if bias_lengths != layer_sizes_forward[1:]:
            weights = [w.T for w in weights]
            layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
        else:
            layer_sizes = layer_sizes_forward
    else:
        layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]

    n_layers = len(layer_sizes)

    # --- Build Graph ---
    G = nx.DiGraph()
    pos = {}
    h_spacing, v_spacing = 4, 1

    for li, size in enumerate(layer_sizes):
        for ni in range(size):
            node = f"L{li}N{ni}"
            pos[node] = (li * h_spacing, -ni * v_spacing)
            G.add_node(node)

    # --- Add edges (skip zero weights) ---
    for i, W in enumerate(weights):
        for j in range(W.shape[0]):
            for k in range(W.shape[1]):
                w = W[j, k]
                if w == 0:  # skip zero weights
                    continue
                src = f"L{i}N{k}"
                dst = f"L{i+1}N{j}"
                G.add_edge(src, dst, weight=w)

    # --- Edge colors: green/red for sign ---
    ecolors = []
    ewidths = []

    # Handle case where all weights might be zero
    if any(abs(w).max() > 0 for w in weights):
        max_abs_w = max(abs(w).max() for w in weights)
    else:
        max_abs_w = 1

    for _, _, d in G.edges(data=True):
        w = d["weight"]
        color = "green" if w > 0 else "red"
        width = 1 + 4 * abs(w) / (max_abs_w + 1e-8)
        ecolors.append(color)
        ewidths.append(width)
        
    # --- Draw ---
    plt.figure(figsize=(h_spacing * n_layers, max(layer_sizes) / 2))
    nx.draw(
        G, pos,
        with_labels=False,
        node_color="lightgray",  # all nodes same neutral color
        node_size=node_radius,
        edge_color=ecolors, width=ewidths, alpha=0.9
    )

    plt.title("Neural Network Visualization (Green=Positive, Red=Negative, No Zero Weights)")
    plt.axis("off")
    plt.show()



def plot_correct_vs_given(correct_outputs, predicted_outputs, accuracy=None, network_name="Network"):
    """
    Plots correct outputs vs predicted outputs for binary/multi-output classification.
    Shows accuracy on the plot if provided.
    """
    correct_outputs = np.array(correct_outputs) 
    correct_L = correct_outputs[:,0]
    correct_R = correct_outputs[:,1]
    predicted_outputs = np.array(predicted_outputs)
    predict_L = correct_outputs[:,0]
    predict_R = correct_outputs[:,1]
    
    plt.figure(figsize=(6, 6),)

    # Flatten outputs for plotting
    plt.scatter(range(correct_outputs.size), correct_outputs.flatten(),
                color="green", alpha=0.7, label="Correct Outputs")
    plt.scatter(range(predicted_outputs.size), predicted_outputs.flatten(),
                color="blue", alpha=0.7, label="Predicted Outputs")

    # Display accuracy on the plot
    if accuracy is not None:
        plt.text(0.95, 0.05, f"Accuracy: {accuracy:.2f}%",
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 transform=plt.gca().transAxes,
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.xlabel("Trial Number")
    plt.ylabel("Output Value")
    plt.title(f"{network_name}: Correct vs Predicted Outputs")
    plt.grid(True)
    plt.legend()
    plt.show()
    
#initalize networks 

net1 = Net (np.array([[-1.56500774, -1.48564045],[ 2.2141425 , -0.1930469 ]]), 
            np.array([[ 1.2748518 , -0.78554613],[-1.88546905,  0.60332706]]), 
            np.array([[2.19992354, 2.28209086],[0.6003428 , 1.94881883]]), 
            np.array([0.08425877, 0.50297257]), 
            np.array([ 2.06062116 ,-1.27320508]), 
            np.array([-2.46655845 ,-2.56175011]))


net2 = Net (np.array([[ 0.1506358 ,  2.89459079],[ 2.28672888, -0.34891209]]), 
            np.array([[ 0.87219453 ,-2.75428546],[ 0.47265078 ,-2.14495651]]), 
            np.array([[ 0.85601805 , 0.61318369],[-2.7341633 , -2.89826288]]), 
            np.array([0.41533157 ,0.18758702]), 
            np.array([0.68160202 ,1.82993832]), 
            np.array([0.82036923, 1.27168403]))

net3 = Net (np.array([[-0.02613776 , 2.71313371],[ 2.47707646 ,-0.06400806]]), 
            np.array([[ 1.09247965 ,-2.43010983],[ 0.44940972 ,-3.11610713]]), 
            np.array([[ 0.33100574 , 0.40940269],[-3.27237737, -2.93965498]]), 
            np.array([0.29752628, 0.24349519]), 
            np.array([0.40626041, 1.73704504]), 
            np.array([1.91571467 ,1.6288842 ]))


def graph_networks():
    net1.graph()
    net2.graph()
    net3.graph()
    
    
def test (I = x_lesion, O = y_lesion):
    loss1, accuracy1, o_preds1, pred_binary1 = net1.forward(I, O)
    loss2, accuracy2, o_preds2, pred_binary2 = net2.forward(I, O)
    loss3, accuracy3, o_preds3, pred_binary3 = net3.forward(I, O)
    #graph of our answers vs the networks 
    plot_correct_vs_given(y_lesion, o_preds1, accuracy1, network_name="Net1")
    plot_correct_vs_given(y_lesion, o_preds2,accuracy2, network_name="Net2")
    plot_correct_vs_given(y_lesion, o_preds3, accuracy3,network_name="Net3")   
    
    


def L_sensory_lesion():
    net1.lesion("L_sensory")
    net2.lesion("L_sensory")
    net3.lesion("L_sensory")
    graph_networks()
    test()
    test(x_L_sensory, y_L_sensory)
    
def R_sensory_lesion():
    net1.lesion("R_sensory")
    net2.lesion("R_sensory")
    net3.lesion("R_sensory")
    graph_networks()
    test()
    test(x_R_sensory, y_R_sensory)
    
#not helpful    
def L_trigeminal_lesion():
    net1.lesion("L_trigeminal")
    net2.lesion("L_trigeminal")
    net3.lesion("L_trigeminal")
    graph_networks()
    test()
#not helpful   
def R_trigeminal_lesion():
    net1.lesion("R_trigeminal")
    net2.lesion("R_trigeminal")
    net3.lesion("R_trigeminal")
    graph_networks()
    test()     
#not helpful    
def L_motor_lesion():
    net1.lesion("L_motor")
    net2.lesion("L_motor")
    net3.lesion("L_motor")
    graph_networks()
    test()   
#not helpful       
def R_motor_lesion():
    net1.lesion("R_motor")
    net2.lesion("R_motor")
    net3.lesion("R_motor")
    graph_networks()
    test()   
    
def L_muscle_lesion():
    net1.lesion("L_muscle")
    net2.lesion("L_muscle")
    net3.lesion("L_muscle")
    graph_networks()
    test()
    test(x_L_muscle, y_L_muscle)    
    
def R_muscle_lesion():
    net1.lesion("R_muscle")
    net2.lesion("R_muscle")
    net3.lesion("R_muscle")
    graph_networks()
    test()
    test(x_R_muscle, y_R_muscle)        