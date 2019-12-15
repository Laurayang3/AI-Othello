import othello as oth
import random as rd
import numpy as np
import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore", category=FutureWarning)

def initialize_game():
	BLANK = "_"
	Board_size = None
	while Board_size == None or Board_size%2 == 1:
		try:
			Board_size = int(input("Enter an even integer to scale the board: "))
			if Board_size%2 == 1:
				print(); print(Board_size,"is an odd number!")
		except ValueError:
			print('\n'"Type Error!")
	
	a_thetacompute = None
	b_thetacompute = None
	while (a_thetacompute == None) or (b_thetacompute == None) or (a_thetacompute + 0.5*b_thetacompute > 1) or (a_thetacompute + 0.5*b_thetacompute < 0) or (a_thetacompute <= 0):
		try:
			a_thetacompute = float(input("Enter 1st value. Ensure flipping possibility (FP=<your first input + 0.5*<your second input>) varies from 0 to 1: "))
			b_thetacompute = float(input("Enter 2nd value. Ensure flipping possibility (FP=<your first input + 0.5*<your second input>) varies from 0 to 1: "))
			if a_thetacompute + 0.5*b_thetacompute > 1 or a_thetacompute + 0.5*b_thetacompute < 0:
				print(); print(a_thetacompute + 0.5*b_thetacompute,"is out of range!")
		except ValueError:
			print('\n'"Type Error!")
	
	Number_shadow_area = None
	while Number_shadow_area == None or Number_shadow_area == 0 or Number_shadow_area > Board_size**2:
		try:
			Number_shadow_area = int(input("Enter an integer to scale the area of shadow: "))
			if Number_shadow_area == 0 or Number_shadow_area > Board_size**2:
				print(); print(Number_shadow_area,"is an out of range!")
		except ValueError:
			print('\n'"Type Error!")

	Area_shadow = []
	start_four = [(Board_size//2, Board_size//2), (Board_size//2-1, Board_size//2), (Board_size//2, Board_size//2-1), (Board_size//2-1, Board_size//2-1)]
	while len(Area_shadow) < Number_shadow_area:
		i = rd.randint(0,Board_size-1); j = rd.randint(0,Board_size-1)
		if (i,j) in Area_shadow or (i,j) in start_four: 
			pass
		else: 
			Area_shadow.append((i,j))

	state_ini = np.array([[BLANK]*Board_size]*Board_size)
	state_ini[Board_size//2-1,Board_size//2-1]="x"
	state_ini[Board_size//2-1,Board_size//2]="o"
	state_ini[Board_size//2,Board_size//2-1]="o"
	state_ini[Board_size//2,Board_size//2]="x"

	return(Board_size, Number_shadow_area, Area_shadow, start_four, state_ini, a_thetacompute, b_thetacompute)

def where_to_play_disk(state,symbol):
	available_step = []
	for i in range(x):
		for j in range(x):
			if state[i,j] == "_":
				validity = False
				validity,flipping_list = oth.move(state, symbol, i, j)
				if validity != False:
					available_step.append((i,j))
	return(available_step)

def next_player(symbol):
	if symbol == "x":
		symbol = "o"
	else: 
		symbol = "x"
	return(symbol)

def learn_theta(state):
	x,y,z = oth.MLE(state)
	a,b = oth.LRG(x,y,z)
	return(a,b)
	# Baby_theta = oth.AItheta(state,a,b)
	# real_theta = oth.thetacompute(state)
	# print(a,b,Baby_theta,real_theta)

def train_othello(state): 
    a,b = learn_theta(state)
    return(a,b)

def trained_othello(state,symbol,a,b):
    nodes = 0
    next_state = state
    # i = 0
    # oth.state_print(next_state)
    while oth.final(next_state,symbol) or oth.final(next_state,next_player(symbol)):
	    if oth.final(next_state,symbol) == False:
		    symbol = next_player(symbol)
		    if symbol == "x":
			    next_step, nodez = oth.expectiminimax(next_state,symbol,1,constant_a,constant_b)
		    else:
		        # Baby_theta = oth.AItheta(next_state,a,b)
		        next_step, nodes = oth.expectiminimax(next_state,symbol,1,a,b)
	    else:
		    if symbol == "x":
			    next_step, nodez = oth.expectiminimax(next_state,symbol,1,constant_a,constant_b)
		    else:
		        # Baby_theta = oth.AItheta(next_state,a,b)
		        next_step, nodes = oth.expectiminimax(next_state,symbol,1,a,b)
	    next_state,f_disks = oth.move(next_state,symbol,next_step[0],next_step[1])
	    # i += 1
	    # if i%5 == 0: 
	    	# print('\n',i," steps were made")
	    	# oth.state_print(next_state)
	    symbol = next_player(symbol)

    Baby_AI_score = oth.score(next_state)
    Teaching_AI_score = 1 - Baby_AI_score
    # print('\n'"Baby-AI score is",Baby_AI_score)
    # print("Teaching-AI score is",Teaching_AI_score)
    if Baby_AI_score > Teaching_AI_score:
	    # print("Congrats, Baby-AI wins!")
	    return(nodes,1,0,Baby_AI_score,Teaching_AI_score)
    elif Baby_AI_score == Teaching_AI_score:
	    # print("What?! Draw?!")
	    return(nodes,0,0,Baby_AI_score,Teaching_AI_score)
    else:
	    # print("Oooooops, Teaching-AI wins!")
	    return(nodes,0,1,Baby_AI_score,Teaching_AI_score)

def untrained_othello(state,symbol):
    nodes = 0
    next_state = state
    # i = 0
    # oth.state_print(next_state)
    while oth.final(next_state,symbol) or oth.final(next_state,next_player(symbol)):
	    if oth.final(next_state,symbol) == False:
		    symbol = next_player(symbol)
		    if symbol == "x":
			    next_step, nodez = oth.expectiminimax(next_state,symbol,1,constant_a,constant_b)
		    else:
		        # Baby_theta = oth.AItheta(next_state,a,b)
		        next_step, nodes = oth.expectiminimax(next_state,symbol,1,1,0)
	    else:
		    if symbol == "x":
			    next_step, nodez = oth.expectiminimax(next_state,symbol,1,constant_a,constant_b)
		    else:
		        # Baby_theta = oth.AItheta(next_state,a,b)
		        next_step, nodes = oth.expectiminimax(next_state,symbol,1,1,0)
	    next_state,f_disks = oth.move(next_state,symbol,next_step[0],next_step[1])
	    # i += 1
	    # if i%5 == 0: 
	    	# print('\n',i," steps were made")
	    	# oth.state_print(next_state)
	    symbol = next_player(symbol)

    Baby_AI_score = 1 - oth.score(next_state)
    Teaching_AI_score = 1 - Baby_AI_score
    # print('\n'"Baby-AI score is",Baby_AI_score)
    # print("Teaching-AI score is",Teaching_AI_score)
    if Baby_AI_score > Teaching_AI_score:
	    # print("Congrats, Baby-AI wins!")
	    return(nodes,1,0,Baby_AI_score,Teaching_AI_score)
    elif Baby_AI_score == Teaching_AI_score:
	    # print("What?! Draw?!")
	    return(nodes,0,0,Baby_AI_score,Teaching_AI_score)
    else:
	    # print("Oooooops, Teaching-AI wins!")
	    return(nodes,0,1,Baby_AI_score,Teaching_AI_score)	    
#--------------------------------------------------------------------------------------------------------------------------------------------------#
x,num_shadow,areaS,start_f,state_init,constant_a,constant_b = initialize_game()
oth.shadow = areaS
oth.N = x
oth.a_true = constant_a
oth.b_true = constant_b
Statistical_wNe = np.array([[None]*5]*30)
u_Statistical_wNe = np.array([[None]*5]*30)
Statistical_aNb = np.array([[None]*2]*25)
print('\n'"max flipping rate is",oth.a_true + 0.5*oth.b_true)
print("Board size is",x,", Number of shadow area is",num_shadow,", Shadow area is",areaS, ", initial four disks are at",start_f)
# print()

print('\n'"untrained AI is playing with Coach AI(an AI that knows all the information) ...")
list1 = []
x_axis = []
for i in range(30):
	u_Statistical_wNe[i,0],u_Statistical_wNe[i,1],u_Statistical_wNe[i,2],u_Statistical_wNe[i,3],u_Statistical_wNe[i,4] = untrained_othello(state_init,"o")
	list1.append(u_Statistical_wNe[i,0])
	x_axis.append(i+1)
plt.suptitle('untrained AI with Cocach AI')
plt.plot(x_axis, list1)    # YMM
plt.ylabel('efficiency')
plt.xlabel('iteration')
plt.show()
print(u_Statistical_wNe)
print("Baby AI won",(np.sum(u_Statistical_wNe,axis=0))[1],"time(s)")
print("Coach AI won",(np.sum(u_Statistical_wNe,axis=0))[2],"time(s)")
print("Baby AI total score is",(np.sum(u_Statistical_wNe,axis=0))[3])
print("Coach AI total score is",(np.sum(u_Statistical_wNe,axis=0))[4])

print('\n'"We are training AI. Please be patient...")
for i in range(25):
	Statistical_aNb[i,0],Statistical_aNb[i,1] = train_othello(state_init)
print(Statistical_aNb)
plt.suptitle('untrained AI with Cocach AI')
plt.plot(x_axis, list1)    # YMM
plt.ylabel('efficiency')
plt.xlabel('iteration')
plt.show()
print("Average of a is", (np.sum(Statistical_aNb,axis=0)/25)[0],"Average of b is",(np.sum(Statistical_aNb,axis=0)/25)[1])
Avg_a = (np.sum(Statistical_aNb,axis=0)/25)[0]
Avg_b = (np.sum(Statistical_aNb,axis=0)/25)[1]
print('\n'"AI is ready!")

print('\n'"trained AI is playing with Coach AI(an AI that knows all the information) ...")
for i in range(30):
	Statistical_wNe[i,0],Statistical_wNe[i,1],Statistical_wNe[i,2],Statistical_wNe[i,3],Statistical_wNe[i,4] = trained_othello(state_init,"o",Avg_a,Avg_b)
	list1.append(Statistical_wNe[i,0])
	x_axis.append(i+1)
plt.suptitle('untrained AI with Cocach AI')
plt.plot(x_axis, list1)    # YMM
plt.ylabel('efficiency')
plt.xlabel('iteration')
plt.show()
print(Statistical_wNe)

print("Baby AI won",(np.sum(Statistical_wNe,axis=0))[1],"time(s)")
print("Coach AI won",(np.sum(Statistical_wNe,axis=0))[2],"time(s)")
print("Baby AI total score is",(np.sum(Statistical_wNe,axis=0))[3])
print("Coach AI total score is",(np.sum(Statistical_wNe,axis=0))[4])
