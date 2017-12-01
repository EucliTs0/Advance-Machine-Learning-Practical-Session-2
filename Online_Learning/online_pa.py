import numpy as np

#Online Passive-Aggressive Algorithms

#Three possible updates:
    #i) Classic Update
    #ii) First Relaxation
    #iii) Second Relaxation
    
#i) Classic Update
def pa_classic(lt, xt):
    return lt / xt

#ii) First Relaxation
def pa_firstRel(C, lt, xt):
    return min(C, (lt / xt))

#iii) Second Relaxation
def pa_secondRel(C, lt, xt):
    return lt / (xt + (1 / (2*C)))

#Passive-Aggressive for binary classification
def pa_binaryClf(train_data, train_labels, iterations, C, update_option):
    #Initialize weights
    L = len(train_data[0])
    #L = data.shape[1]
    w = np.zeros(L)
    #Algorithm
    for i in xrange(iterations):
        for index in xrange(len(train_data)):
        
            xt = train_data[index, :]
            y = train_labels[index] * np.dot(w, xt)
            if y < 1:
                #compute loss
                lt = max(0, (1 - y))
                xt_square = np.sum(xt ** 2)
                
                #Select one of the updates
                
                if (update_option == 'classic'):
                    tt = pa_classic(lt, xt_square)
                elif (update_option == 'first'):
                    tt = pa_firstRel(C, lt, xt_square)
                else:
                    tt = pa_secondRel(C, lt, xt_square)
                    
                                                
                #update weights
                w = w + tt * train_labels[index] * xt
        return w
    
    
    
    

            
        
                
            

