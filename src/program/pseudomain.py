#pseudo main file that runs of the execture main
import random
class Main:
    
    
    def __init__(self):
        tree = DepTree(10,4)
        tree.randompath()
    
    

class DepTree:
    
    def __init__(self,d,c):
        self.d = d
        self.c = c
        
        
    def classes(self):
        pass
    def randompath(self):
        rand = random.Random()
        rand.seed(100897337)
        
        for i in range(0,5):
            if(rand.uniform(0,1) >0.5):
                print("Left")
            else:
                print("Right")
        
        


