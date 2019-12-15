import numpy as np
import copy as cp
import random as rd

BLANK = "_"
N=4
shadow=[]
a_true=0.5
b_true=0.8
node_number=0

def thetacompute(state): #compute the real theta
    o=0;
    x=0;
    for i in range(N):
        for j in range(N):
            if (state[i,j]=="x"):
                x+=1
            if (state[i,j])=="o":
                o+=1
    if (x==0 or o==0): return a_true
    theta=a_true+b_true/(o*1./x+x*1./o)
    return theta

def AItheta(state,a,b): #the theta AI thinks
    o=0;
    x=0;
    for i in range(N):
        for j in range(N):
            if (state[i,j]=="x"):
                x+=1
            if (state[i,j])=="o":
                o+=1
    if (x==0 or o==0): return a
    theta=a+b/(o*1./x+x*1./o)
    return theta

def __MLEmove(state,symbol, row, col): #return False if the move is invalid, else return the new state,
                                       #for MLE training only since we use shadow=whole board when training to simplify the process
    shadow=[]
    for i in range(N):
        for j in range(N):
            shadow.append((i,j))
    
    if state[row,col] != BLANK: return False, False
    valid=False
    oppo="x" if symbol == "o" else "o"
    rdir=[-1,0,1]
    cdir=[-1,0,1]
    final_dir=[]
    change=[]
    for i in rdir:
        for j in cdir:
            r=row+i
            c=col+j
            while (r<N and c<N and c>-1 and r>-1 and state[r,c]==oppo):
                r+=i
                c+=j
            if ((r!=row+i or c!=col+j) and r<N and c<N and c>-1 and r>-1 and state[r,c]==symbol): #meaning stop at a position of symbol and has at least 1 opponent's chess between
                valid=True
                final_dir.append((i,j))
                          
    if (valid==True): #find the chess need to be changed
        new_state = cp.deepcopy(state)
        for tdir in final_dir:
            r=row+tdir[0]
            c=col+tdir[1]
            while(state[r,c]==oppo):
                change.append((r,c))
                r+=tdir[0]
                c+=tdir[1]
    else:
        return False,False

    if ((row,col) in shadow): #compute the probablity for each chess to change successfully
        for i in change:
            x=rd.random()
            theta=thetacompute(state)
            if x<theta:
                new_state[i[0],i[1]]=symbol
        new_state[row,col]=symbol
    else:
        for i in change:
            new_state[i[0],i[1]]=symbol
        new_state[row,col]=symbol
    return new_state,change

def one_move(state,symbol):
    for i in range(N):
        for j in range(N):
            child,moveable=__MLEmove(state,symbol,i,j)
            if child is False: continue
            return i,j
    return -1,-1
            
def MLE(state):
    repeattime=500
    symbol="x"
    thetalist=[]
    olist=[]
    xlist=[]
    for step in range(N*N-2): # the number of theta-nodes pairs
        row,col=one_move(state,symbol)
        flipsum=0
        if row==-1: break
        for i in range(repeattime):
            child,moveable=__MLEmove(state,symbol,row,col)
            for j in range(len(moveable)):
                if child[moveable[j][0],moveable[j][1]]==symbol:
                    flipsum+=1
        total=len(moveable)*repeattime
        thetalist.append(flipsum*1./total)
        o=0
        x=0
        for i in range(N):
            for j in range(N):
                if (state[i,j]=="x"):
                    x+=1
                elif (state[i,j]=="o"):
                    o+=1
        olist.append(o)
        xlist.append(x)
        state=child
        symbol="x" if symbol == "o" else "o"    
    return thetalist, olist, xlist

def LRG(y,x1,x2): #do linear regression to find the relationship between the theta and the number of x and o on the board
    x=[]
    rate=0.001
    for i in range(len(x1)):
        if x1[i]==0 or x2[i]==0:
            x.append(0)
        else:
            x.append(1./((x1[i]*1./x2[i])+x2[i]*1./x1[i]))
    a=0
    b=0
    x=np.array(x)
    f=a+b*x
    e=sum((y-f)*(y-f))
    while(e>0.01):
        a+=2*rate*sum(y-f)
        b+=2*rate*sum((y-f)*x)
        f=a+b*x
        e=sum((y-f)*(y-f))
    return a,b


def state_print(state): #print the current board
    print("   ",end='')
    for i in range(N):
        print(i," ",end='')
    print()
    for i in range(N):
        print (i," ",end='')
        for j in range(N):
            print(state[i,j]," ",end='')
        print()
        
def score(state):#return the portion of "x"
    x=0
    o=0
    for i in range(N):
        for j in range(N):
            if (state[i,j]=="x"):
                x+=1
            elif (state[i,j]=="o"):
                o+=1
    return x*1./(x+o)                                       

def final(state,symbol):
    f=False
    for i in range(N):
        for j in range(N):
            child,moveable=move(state,symbol, i,j)
            if not (child is False):
                f=True   
    return f

def move(state,symbol, row, col): #return False if the move is invalid, else return the new state, if the move is in the shadow, possible to fail to change some chess
    if state[row,col] != BLANK: return False, False
    valid=False
    oppo="x" if symbol == "o" else "o"
    rdir=[-1,0,1]
    cdir=[-1,0,1]
    final_dir=[]
    change=[]
    for i in rdir:
        for j in cdir:
            r=row+i
            c=col+j
            while (r<N and c<N and c>-1 and r>-1 and state[r,c]==oppo):
                r+=i
                c+=j
            if ((r!=row+i or c!=col+j) and r<N and c<N and c>-1 and r>-1 and state[r,c]==symbol): #meaning stop at a position of symbol and has at least 1 opponent's chess between
                valid=True
                final_dir.append((i,j))
                          
    if (valid==True): #find the chess need to be changed
        new_state = cp.deepcopy(state)
        for tdir in final_dir:
            r=row+tdir[0]
            c=col+tdir[1]
            while(state[r,c]==oppo):
                change.append((r,c))
                r+=tdir[0]
                c+=tdir[1]
    else:
        return False,False

    if ((row,col) in shadow): #compute the probablity for each chess to change successfully
        theta=thetacompute(state)
        for i in change:
            x=rd.random()
            if x<theta:
                new_state[i[0],i[1]]=symbol
        new_state[row,col]=symbol
    else:
        for i in change:
            new_state[i[0],i[1]]=symbol
        new_state[row,col]=symbol
    return new_state,change


            
def expectiminimax(state,symbol,layer,a,b):
    global node_number
    theta1=AItheta(state,a,b)
    if layer<1 or layer>6: return False
    if layer==6:
        node_number+=1
        return score(state)
    location=[]
    expecti=[]
    oppo="x" if symbol == "o" else "o"
    for i in range(N):
        for j in range(N):
            child, moveable=move(state,symbol,i,j)
            if child is False: continue
            location.append((i,j))
            #oppo="x" if symbol == "o" else "o"
            if (i,j) in shadow:
               
                control=np.zeros(len(moveable))
                expectation=0 #the expected value for this move
                new_state = cp.deepcopy(state)
                new_state[i,j]=symbol
                expectation+=expectiminimax(new_state,oppo,layer+1,a,b)*((1-theta1)**len(moveable))
                for k in range(2**len(moveable)-1):
                    new_state = cp.deepcopy(state)
                    new_state[i,j]=symbol
                    p=0
                    while (control[p]==1): p+=1
                    control[p]=1
                    for q in range(p):
                        control[q]=0
                    m=0
                    for q in range(len(moveable)):
                        if control[q]==1:
                            m+=1
                            new_state[moveable[q][0],moveable[q][1]]=symbol
                    prob=(theta1**m)*((1-theta1)**(len(moveable)-m))
                    expectation+=prob*expectiminimax(new_state,oppo,layer+1,a,b)
                expecti.append(expectation)
            else:
                expecti.append(expectiminimax(child,oppo,layer+1,a,b))
            
    if len(expecti)==0:
        if final(state,oppo): # both player cannot move
            return expectiminimax(state,oppo,layer,a,b) #give the turn to opponent
        else:
            node_number+=1
            return score(state)

    if (symbol=="x"):
        max=expecti[0]
        maxloc=0
        
        for i in range(len(location)):
            if expecti[i]>max:
                maxloc=i
                max=expecti[i]
        if (layer==1):
            # print(location)
            # print(expecti)
            return location[maxloc],node_number
        else:
            return max
    else:
        min=expecti[0]
        minloc=0
        for i in range(len(location)):
            if expecti[i]<min:
                minloc=i
                min=expecti[i]
        if (layer==1):
            # print(location)
            # print(expecti)
            return location[minloc],node_number
        else:
            return min
    

if __name__=="__main__":
    shadow.append((0,0))
    state0=np.array([[BLANK]*N]*N)
    state0[0,0]="x"
    state0[0,1]="x"
    state0[0,2]="x"
    state0[2,2]="x"
    state0[0,3]="o"
    state0[1,0]="o"
    state0[1,1]="o"
    state0[1,2]="o"
    #state0[2,1]="o"
    print(thetacompute(state0))
    state_print(state0)
    
    
    
