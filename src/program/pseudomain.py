#pseudo main file that runs of the execture main
import random
import math
import copy
class Main:
    
    
    def __init__(self):
        tree = DepTree(10,4)
        tree.init_random()
        tree.init_features()
        tree.gen_tree_v2()
        tree.init_classes()
        tree.gen_class_to_feature_value_based_on_deptree()
        tree.generate_data_sheet()
        training = TrainAndTest(tree.data_sheet,tree.rand)
        training.shuffle_trainingset()
        training.k_fold(8000)
class DepTree:
    
    def __init__(self,d,c):
        self.d = d
        self.c = c

    def init_random(self):
        self.rand = random.Random()
        self.rand.seed(7337)
    
    def assign_classification_value(self):
        pass
    
    def gen_tree_v2(self):
        state1 = [0]
        state1[0] = self.feature_set[0]
        self.dep_tree = Tree(state1)
        for i in range(1,len(self.feature_set)):
            temp_state = [0]
            temp_state[0] = self.feature_set[i]
            r_val = self.rand.randint(0,100)
            r_val = r_val /100.0
            if r_val < 0.35 or len(self.dep_tree.get_leaf_nodes(self.dep_tree.baseroot)) < 1: # add to root
                self.dep_tree.add_node(TreeNode(temp_state),self.dep_tree.baseroot)
            elif r_val > 0.35 and r_val < 0.60: #add to leaf node of base
                index = self.rand.randint(0,len(self.dep_tree.get_leaf_nodes(self.dep_tree.baseroot))-1)
                self.dep_tree.add_node(TreeNode(temp_state),self.dep_tree.get_leaf_nodes(self.dep_tree.baseroot)[index])
            else: # add to as leaf of previous state
                prev_state = [0]
                prev_state[0] =self.feature_set[i-1]
                self.dep_tree.add_node(TreeNode(temp_state),self.dep_tree.get_node_of_state(prev_state))
                
        print(self.dep_tree)
        
    def init_features(self):
        self.feature_set = [] 
        self.feature_edges = []
    
        for i in range(0,self.d):
            name_val = "F" + str(i)
            self.feature_set.append(name_val)
    def init_classes(self):
        self.class_set = []
        for i in range(0,self.c):
            self.class_set.append("C"+str(i))
            
    
    def gen_class_to_feature_value_based_on_deptree(self):
        self.feat_set_value = []
        
        top_layer_set = []
        for c in self.class_set: # generate random value for the 4 classes when F0 is 0
            r_val = self.rand.randint(0,100)
            r_val = r_val/100.0
            name = "F0," + c + ":" + str(r_val)
            top_layer_set.append(name)
        self.feat_set_value.append(top_layer_set)
        for f in self.feature_set:
            statef = [f]
            currnode = self.dep_tree.get_node_of_state(statef)
            temp_set = []
            if not currnode.root == None:
                rootstate= currnode.root.state               
                for c in self.class_set:
                    r_val1 = self.rand.randint(0,100) /100.0 # given root node is 0 and curr node = 0
                    r_val2 = self.rand.randint(0,100) /100.0 # given root node is 1 and curr node = 0
                    name = currnode.state[0] + "|" + rootstate[0] + "," + c + ":" + str(r_val1) + "-" + str(r_val2)
                    temp_set.append(name)
                self.feat_set_value.append(temp_set)
        for setf in self.feat_set_value:
            print(setf)
    
    def generate_data_sheet(self):
        self.data_sheet = []
        for c in range(0,len(self.class_set)):
            for i in range(0,2000):
                F0_Threshold =  float(self.feat_set_value[0][c].split(":")[1]) 
                F0  = self.rand.randint(0,100)/100.0 > F0_Threshold
                if F0:
                    F1_Threshold =  float(self.feat_set_value[1][c].split(":")[1].split("-")[1])
                    F8_Threshold =  float(self.feat_set_value[8][c].split(":")[1].split("-")[1])
                    F9_Threshold =  float(self.feat_set_value[9][c].split(":")[1].split("-")[1])
                    
                else:
                    F1_Threshold =  float(self.feat_set_value[1][c].split(":")[1].split("-")[0])
                    F8_Threshold =  float(self.feat_set_value[8][c].split(":")[1].split("-")[0])
                    F9_Threshold =  float(self.feat_set_value[9][c].split(":")[1].split("-")[0])
                F1  = self.rand.randint(0,100)/100.0 > F1_Threshold
                F8  = self.rand.randint(0,100)/100.0 > F8_Threshold
                F9  = self.rand.randint(0,100)/100.0 > F9_Threshold                
                if F1:
                    F2_Threshold =  float(self.feat_set_value[2][c].split(":")[1].split("-")[1])
                    F3_Threshold =  float(self.feat_set_value[3][c].split(":")[1].split("-")[1])
                    F6_Threshold =  float(self.feat_set_value[6][c].split(":")[1].split("-")[1])                   
                else:
                    F2_Threshold =  float(self.feat_set_value[2][c].split(":")[1].split("-")[0])
                    F3_Threshold =  float(self.feat_set_value[3][c].split(":")[1].split("-")[0])
                    F6_Threshold =  float(self.feat_set_value[6][c].split(":")[1].split("-")[0])
                F2  = self.rand.randint(0,100)/100.0 > F2_Threshold
                F3  = self.rand.randint(0,100)/100.0 > F3_Threshold
                F6  = self.rand.randint(0,100)/100.0 > F6_Threshold                
                
                if F3:
                    F4_Threshold =  float(self.feat_set_value[4][c].split(":")[1].split("-")[1])
                else:
                    F4_Threshold =  float(self.feat_set_value[4][c].split(":")[1].split("-")[0])
                F4  = self.rand.randint(0,100)/100.0 > F4_Threshold
                if F4:
                    F5_Threshold =  float(self.feat_set_value[5][c].split(":")[1].split("-")[1])
                else:
                    F5_Threshold =  float(self.feat_set_value[5][c].split(":")[1].split("-")[0])                   
                if F6:
                    F7_Threshold =  float(self.feat_set_value[7][c].split(":")[1].split("-")[1])
                else:
                    F7_Threshold =  float(self.feat_set_value[7][c].split(":")[1].split("-")[0])                    
                F5  = self.rand.randint(0,100)/100.0 > F5_Threshold
                F7  = self.rand.randint(0,100)/100.0 > F7_Threshold                
                
                temp_set =[c,int(F0),int(F1),int(F2),int(F3),int(F4),int(F5),int(F6),int(F7),int(F8),int(F9)]
                print(temp_set)
                self.data_sheet.append(temp_set)
                
    def get_feature_value(feature):
        feat_sep = feature.split(":")
        val = float(feat_sep[1])
        name = feat_sep[0]
        
        return name,val
    def get_propablilty_by_class(array):
        pass
class TreeNode:

    def __init__(self,state,root=None):
        self.root = root
        self.state = state
    def get_state(self):
        return self.state
    def get_move(self):
        return self.state[4]
    def get_root(self):
        return self.root

    def get_height(self,height=0):
        if(self.root == None or self.root == 0):
            return height
        else:
            height +=1
            return self.root.get_height(height)

class Tree:
    def __init__(self,init_state):
        self.baseroot = TreeNode(init_state)
        self.nodes = [self.baseroot]
    def add_node(self,node,root=None):
        if(root == None):
            node.root = self.baseroot
        else:
            node.root = root
            self.nodes.append(node)
    def get_nodes_of_level(self,level):
        l_nodes = []
        for node in self.nodes:
            if node.get_height() == level:
                l_nodes.append(node)
        return l_nodes
    def get_leaf_nodes(self,bnode=None):
        if bnode == None:
            bnode = self.baseroot
        l_nodes = []
        for node in self.nodes:
            if(node.root == bnode):
                l_nodes.append(node)
        return l_nodes
    def get_node_of_state(self,state):
        for node in self.nodes:
            if node.state == state:
                return node

        print("What?")
        return 0
    def __str__(self):
        tstr = ""
        not_deepest_reach = True
        level = 0
        while(not_deepest_reach):
            not_deepest_reach = False
            for node in self.nodes:
                if node.get_height() == level:
                    not_deepest_reach = True
                    if node.root == None:
                        root_str = "None"
                    else:
                        root_str =str(node.root.state[0])
                    tstr += "State: " + str(node.state) + " base: " + root_str  + "\n"
              
            level = level + 1
    
        return tstr


class TrainAndTest:
    
    
    def __init__(self,data_sheet,rand):
        self.training_set = copy.deepcopy(data_sheet)
        self.rand = rand
    def shuffle_trainingset(self):
        self.rand.shuffle(self.training_set)
    
    def k_fold(self,sample_size):
        test_set = []
        train_set = []
        set_size = int(sample_size/5)
        
        for i in range(1,2):
            test_set = []
            train_set = []            
            low = (i-1)*set_size
            high = (i*set_size)
            bot_set_low = 0                        
            bot_set_high= low
            top_set_low = high
            top_set_high = sample_size               
       
            
            test_set = self.training_set[low:high]  
            if bot_set_high > 0:
                train_set = train_set + self.training_set[bot_set_low:bot_set_high]
            if top_set_low < sample_size:
                train_set = train_set + self.training_set[top_set_low:top_set_high]
            
 
            prob_test = TrainAndTest.gather_data_dep_tree(test_set)
            prob_train = TrainAndTest.gather_data_dep_tree(train_set)
            
            
        
    
    def gather_data_dep_tree(t_set):
        probability_t = []
        c0_t = []
        c1_t = []
        c2_t = []
        c3_t = []
        for c in range(0,len(t_set[0])):
            c0_t.append(0)
            c1_t.append(0)
            c2_t.append(0)
            c3_t.append(0)
        
        probability_t = [c0_t,c1_t,c2_t,c3_t]
        for s in t_set:
            c_val = s[0]
            probability_t[c_val][0] =  probability_t[c_val][0] + 1            
            for i in range(1,len(s)):
                
                probability_t[c_val][i] =  probability_t[c_val][i] + s[i]
        return probability_t
   
    def deptreeestimate():
        pass
    
    def bayseian_indepented(test,train):
        p_c0_test = test[0][0]/1600
        p_c1_test = test[1][0]/1600
        p_c2_test = test[2][0]/1600
        p_c3_test = test[3][0]/1600
        
        p_c0_train = train[0][0]/6400
        p_c1_train = train[1][0]/6400
        p_c2_train = train[2][0]/6400
        p_c3_train = train[3][0]/6400
        
        
    
        
        
    def bayseian_depented():
        pass
    
    def desicionstree():
        pass
    