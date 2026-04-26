import random
import numpy as np

y_list = []

#create stimulated and non-stimulated trials
for c in range(800):
        y_list.append([0,0])
        y_list.append([1,1])
        

random.shuffle(y_list)

#add in random ones for generalization 
for d in range(80):
        y_list.append([0,1])
        y_list.append([1,0])

#Create test and train data 
y_test = y_list[:100]
y_train = y_list[100:]

random.shuffle(y_train)

x_train = []
x_test = []

options = ["L", "R", "B"]

for i, l in enumerate(y_train):
        s = sum(l)
        #no blink [0,0], due to neither eye being stimulated 
        if s == 0:
                x_train.append([random.uniform(-1,0), random.uniform(-1,0)])
        #both blink [1,1], due to either or both eyes being stimulated 
        elif s == 2:
                region = options[i % 3]  # evenly distributed
                if region == "L":
                        x_train.append([random.uniform(0.0001,1), random.uniform(-1,0)])
                elif region == "R":
                        x_train.append([random.uniform(-1,0), random.uniform(0.0001,1)])
                else:
                        x_train.append([random.uniform(0.0001,1), random.uniform(0.0001,1)])
        else:  #one blinks [0,1] or [1,0], random for generalization
                x_train.append([random.uniform(-1,1), random.uniform(-1,1)])

   # ---- Testing ---- repeating above but to create test data 
for i, p in enumerate(y_test):
        s = sum(p)
        if s == 0:
                x_test.append([random.uniform(-1,0), random.uniform(-1,0)])
        elif s == 2:
                region = options[i % 3]
                if region == "L":
                        x_test.append([random.uniform(0.0001,1), random.uniform(-1,0)])
                elif region == "R":
                        x_test.append([random.uniform(-1,0), random.uniform(0.0001,1)])
                else:
                        x_test.append([random.uniform(0.0001,1), random.uniform(0.0001,1)])
        else:
                x_test.append([random.uniform(-1,1), random.uniform(-1,1)])

x_train = np.array(x_train) 
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


#Create Lesion Data 

def create_lesion_data():
        random.seed(10)
        y_lesion = []

        for f in range(50):
                y_lesion.append([0,0])
                y_lesion.append([1,1])
        
        random.shuffle(y_lesion)

        x_lesion = []

        options = ["L", "R", "B"]

        for w, q in enumerate(y_lesion):
                s = sum(q)
        #no blink [0,0], due to neither eye being stimulated 
                if s == 0:
                        x_lesion.append([random.uniform(-1,0), random.uniform(-1,0)])
        #both blink [1,1], due to either or both eyes being stimulated 
                elif s == 2:
                        region = options[w % 3]  # evenly distributed
                        if region == "L":
                                x_lesion.append([random.uniform(0.0001,1), random.uniform(-1,0)])
                        elif region == "R":
                                x_lesion.append([random.uniform(-1,0), random.uniform(0.0001,1)])
                        else:
                                x_lesion.append([random.uniform(0.0001,1), random.uniform(0.0001,1)])
        
                
        y_lesion = np.array(y_lesion)
        x_lesion = np.array(x_lesion)
        return y_lesion, x_lesion

y_lesion, x_lesion = create_lesion_data()
        
    
    
# test Data for if the L_sensory_lesion 

def create_L_sensory_data():
        random.seed(10)
        y_L_sensory = []

        for t in range(50):
                y_L_sensory.append([0,0])
                y_L_sensory.append([1,1])
               
        
        
        random.shuffle(y_L_sensory)

        x_L_sensory = []

        options = ["A", "B"]

        for w, q in enumerate(y_lesion):
                s = sum(q)
#no blink [0,0], due to neither eye being stimulated 
                if s == 0:
                        region = options[w % 2]
                        if region == "A":
                                x_L_sensory.append([random.uniform(-1,0), random.uniform(-1,0)])
                        else: 
                                x_L_sensory.append([random.uniform(0.0001,1), random.uniform(-1,0)])
                                
#both blink [1,1], due to either or both eyes being stimulated 
                elif s == 2:
                        region = options[w % 2]  # evenly distributed
                        if region == "A":
                                x_L_sensory.append([random.uniform(0.0001,1), random.uniform(0.0001,1)])
                        else:
                                x_L_sensory.append([random.uniform(-1,0), random.uniform(0.0001,1)])
                   
        return np.array(y_L_sensory), np.array(x_L_sensory)

y_L_sensory, x_L_sensory = create_L_sensory_data()


def create_R_sensory_data():
        random.seed(10)
        y_R_sensory = []

        for t in range(50):
                y_R_sensory.append([0,0])
                y_R_sensory.append([1,1])
                
        random.shuffle(y_L_sensory)

        x_R_sensory = []

        options = ["A", "B"]

        for w, q in enumerate(y_lesion):
                s = sum(q)
#no blink [0,0], due to neither eye being stimulated 
                if s == 0:
                        region = options[w % 2]
                        if region == "A":
                                x_R_sensory.append([random.uniform(-1,0), random.uniform(-1,0)])
                        else: 
                                x_R_sensory.append([random.uniform(-1,0), random.uniform(0.0001,1)])
                                
                                
#both blink [1,1], due to either or both eyes being stimulated 
                elif s == 2:
                        region = options[w % 2]  # evenly distributed
                        if region == "A":
                                x_R_sensory.append([random.uniform(0.0001,1), random.uniform(0.0001,1)])
                        else:
                                x_R_sensory.append([random.uniform(0.0001,1), random.uniform(-1,0)])
                   
        return np.array(y_R_sensory), np.array(x_R_sensory)

y_R_sensory, x_R_sensory = create_R_sensory_data()

def create_L_muscle_data():
        random.seed(10)
        y_L_muscle = []

        for t in range(50):
                y_L_muscle.append([0,0])
                y_L_muscle.append([0,1])
                
        random.shuffle(y_L_muscle)

        x_L_muscle = []
        
        options = ["A", "B", "C"]

        for w, q in enumerate(y_L_muscle):
                s = sum(q)
        #no blink [0,0], due to neither eye being stimulated 
                if s == 0:
                        x_L_muscle.append([random.uniform(-1,0), random.uniform(-1,0)])
        #both blink [1,1], due to either or both eyes being stimulated 
                elif s == 1:
                        region = options[w % 3]  # evenly distributed
                        if region == "A":
                                x_L_muscle.append([random.uniform(0.0001,1), random.uniform(-1,0)])
                        elif region == "B":
                                x_L_muscle.append([random.uniform(-1,0), random.uniform(0.0001,1)])
                        else:
                                x_L_muscle.append([random.uniform(0.0001,1), random.uniform(0.0001,1)])        
        
        return np.array(y_L_muscle), np.array(x_L_muscle)
                        
y_L_muscle, x_L_muscle = create_L_muscle_data()        

def create_R_muscle_data():
        random.seed(10)
        y_R_muscle = []

        for t in range(50):
                y_R_muscle.append([0,0])
                y_R_muscle.append([1,0])
                
        random.shuffle(y_L_muscle)

        x_R_muscle = []
        
        options = ["A", "B", "C"]

        for w, q in enumerate(y_R_muscle):
                s = sum(q)
        #no blink [0,0], due to neither eye being stimulated 
                if s == 0:
                        x_R_muscle.append([random.uniform(-1,0), random.uniform(-1,0)])
        #both blink [1,1], due to either or both eyes being stimulated 
                elif s == 1:
                        region = options[w % 3]  # evenly distributed
                        if region == "A":
                                x_R_muscle.append([random.uniform(0.0001,1), random.uniform(-1,0)])
                        elif region == "B":
                                x_R_muscle.append([random.uniform(-1,0), random.uniform(0.0001,1)])
                        else:
                                x_R_muscle.append([random.uniform(0.0001,1), random.uniform(0.0001,1)])        
        
        return np.array(y_R_muscle), np.array(x_R_muscle)
                        
y_R_muscle, x_R_muscle = create_R_muscle_data()        