# Author: Santhosh Ravichandran
# Description: Python code using the Q-learning algorithm (A reinforcement learning algo) to control a pole on a moving cart.  

if True:
    import numpy as np
    from numpy import sin, cos
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integrate
    import matplotlib.animation as animation
    import pandas as pd

    m1 = 1 #kg, mass of the cart
    m2 = 0.1 # kg, mass of the pole
    L = 1 # m, length of the pole
    g = 9.81 # m/s2, acceleration due to gravity

    totalEpisodes = 50000 # The total number of episodes for training

    rotDamping = 0.01 # a damping coefficient for the pole's rotation. The damping terms are added for stability
    linDamping = 0.1 # damping coefficient for the cart's movement

    a = 0 # acceleration
    omega = 0 # angular velocity of the pole.
    timeElapsed = 0 # time elapsed without failure in an episode
    oldV = 0 
    currV = 0

    Qval = 0 # This is the main value for the reinforcement algorithm to know the value of a state

    dt = 1/30 # time step for the simulation

    angle = 88*(np.pi/180) # The angle of the pole from the horizontal, measured anticlockwise
    print(angle) 
    x = 0.4 # The initial position of the cart. This is getting set in the episode anyway.

    world = pd.DataFrame(columns=['omega','x','angle','v_a','Q'],dtype=np.float32) # This is the main database wariable that holds all the states and their values.

    world = world._append({'omega':0,'x':0,'angle':round(angle,1),'v_a':0,'Q':0},ignore_index=True) # Just appending a state
    
    # state must include velocity of the masses - linear and angular, position of the masses 
    # actions must include only linear velocity of mass 1

    allActions = [-2,0,2] # The set of actions, -2 means left, 0 means hold and 2 means right

    def model(x,angle,oldV,currV1,omega,timeElapsed):
        '''This the function that simulates the physics of the system'''
        x = currV*dt + x # Just a simple integration to find the current position of the pole based on the current velocity
        a = (currV1 - oldV)/dt # The acceleration is found 

        alpha = (3*(-m2*g*L*0.5*np.cos(angle)+m2*a*L*0.5*np.sin(angle)))/(m2*L**2) # This is the angular acceleration of the pole
        omega = alpha*dt + omega # angular velocity integration
        theta = omega*dt + angle # angle integration

        currV1 = currV1 - linDamping*currV1 # applying damping
        omega = omega - rotDamping*omega # applying damping

        timeElapsed = timeElapsed + dt 

        '''Limiting omega to avoid too many states. All we need to '''
        if omega > 1:
            omega = 1
        elif omega < -1:
            omega = -1

        return x,theta,omega,timeElapsed

    def policy(world,currV,omega,x,angle):
        '''This is the function that defines the action based on a state'''

        global allActions,episodes

        tempDF = world[ (world['omega']==round(omega,0)) & (world['angle']==round(angle,1))& (world['x']==round(x,1))] # Just taking the rows of the current state
        actions = tempDF.v_a.values
        values = tempDF.Q.values

        try:
            action = allActions[np.argmax(np.random.rand(3))] # Taking a random action just as a default value in case if the following code doesnt take any action, this is redundant
            if len(world)<450: # Up to 450 states, the action taken is forced to take a random action. This improves knowledge for the agent.
                availableActions = tempDF.v_a.values # extracting the actions that were taken before
                while action in availableActions and len(availableActions)<3:
                    availableActions = tempDF.v_a.values
                    action = allActions[np.argmax(np.random.rand(3))]
                availableActions = tempDF.v_a.values
                #print('state = ', str(tempDF.omega.values[-1])+","+str(tempDF.angle.values[-1])+","+str(tempDF.x.values[-1])," ",'available Actions = ',availableActions,'taking action=',action)
            elif episodes<totalEpisodes and np.random.rand() < 0.3: # The episodes<totalEpisodes can be used to adjust until which episode you want the agent to explore. 0.1 says that 10% of the time, the agent will take random actions
                # print('taking random action!!!!')
                action = allActions[np.argmax(np.random.rand(3))] 
            else:
                action = actions[np.argmax(values)] # This is the action with the highest value.
        except:
            action = allActions[np.argmax(np.random.rand(3))] # If there is an error, just take a random action
            print('error, taking random action')

        return int(action)

    def animate(i):
        """performs one timestep"""

        global x,angle, omega, timeElapsed, currV, oldV, world, Qval

        fail = 0

        if angle > 60*(np.pi/180) and angle < 120*(np.pi/180) and x < 0.5 and x > -0.5:

            QS = world[(world['v_a']==currV) & (world['omega']==round(omega,0)) & (world['x']==round(x,1)) & (world['angle']==round(angle,1))].Q.values # extracting the row for the current state

            currV = policy(world,currV,omega,x,angle) # The action that we take is actually the velocity of the cart, and therefore assigned to currV
            indexes = (world[(world['v_a']==currV) & (world['omega']==round(omega,0)) & (world['x']==round(x,1)) & (world['angle']==round(angle,1))].Q.index) # collecting indexes of states

            old_x = x 
            old_angle = angle
            old_omega = omega

            x, angle,omega,timeElapsed = model(x,angle,oldV,currV,omega,timeElapsed)

            '''Assigning rewards'''
            if (angle < 93*(np.pi/180) and angle > 87*(np.pi/180)) and (x < 0.2 and x > -0.2):
                R = 1
            else:
                R = 0

            QSDash = world[(world['omega']==round(omega,0)) & (world['x']==round(x,1)) & (world['angle']==round(angle,1))].Q.values # The values of the states alone, without actions

            '''This is the Q-learning algo'''
            try:
                Qmax = np.max(QSDash)
                Qval = QS[0] + 0.1*(R+0.9*Qmax-QS[0])
            except:
                Qval = 0 # Just assign a zero if there was an error

            # Put the caclculated Qval in the appropriate location in the database
            if len(indexes) >= 1:
                temp = world[(world['v_a']==currV) & (world['omega']==round(old_omega,0)) & (world['x']==round(old_x,1)) & (world['angle']==round(old_angle,1))].Q.index
                world.at[temp[0],'Q'] = Qval 
            else:
                #print("------- _appending State ---------",round(omega,0),round(angle,1),round(x,1))
                world = world._append({'omega':round(old_omega,0),'x':round(old_x,1),'angle':round(old_angle,1),'v_a':round(currV,1),'Q':Qval},ignore_index=True)

            oldV = currV

        else:
            fail = 1

        return fail,timeElapsed

    # choose the interval based on dt and the time to animate one step
    from time import time
    animate(0)
    episodes = 0
    failTimes = 0
    totalFailTime = 0

    while episodes<totalEpisodes:
        episodes+=1
        x = round(np.random.uniform(-0.25, 0.25),1)
        angle = 88*(np.pi/180) # the initial angle, change this and experiment
        omega = 0 
        timeElapsed = 0
        currV = 0
        oldV = 0  
        for i in range(0,100): # Run until the episode fails, that is, the pole exceeds a certain angle
            fail, timeElapsed = animate(i)
            if fail: 
                if episodes %100 ==0:
                    print('At episode = ', episodes,'Failed at time = ',timeElapsed,'size of world = ',len(world),'average time of failure = ',failTimes)
                totalFailTime += timeElapsed
                failTimes = totalFailTime/episodes
                break
                
'''The following block is to use the learned parameters and use it to actually run and demonstrate the agent's ability to control. Since the same logic is used, comments are not provided'''

if True:
    import numpy as np
    from numpy import sin, cos
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integrate
    import matplotlib.animation as animation
    import pandas as pd

    angle = 88*(np.pi/180)
    print(angle)
    x = 0

    allActions = [-2,0,2]

    #set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-1, 1), ylim=(-2, 5))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        """initialize animation"""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def policy(world,currV,omega,x,angle):

        global allActions

        tempDF = world[ (world['omega']==round(omega,0)) & (world['angle']==round(angle,1))& (world['x']==round(x,1))]
        actions = tempDF.v_a.values
        values = tempDF.Q.values

        try:
            action = actions[np.argmax(values)]
        except:
            print("Error")
            action = allActions[np.argmax(np.random.rand(3))]
        #print('Returning action=',action)
        return int(action)
    
    def policyWA(world,currV,omega,x,angle):

        global allActions

        tempDF = world[ (world['omega']==round(omega,0)) & (world['angle']==round(angle,1))& (world['x']==round(x,1))]
        actions = tempDF.v_a.values
        values = tempDF.Q.values
        
        approximation = 0
        
        try:
            action = actions[np.argmax(values)]
        except:
            print("Error")
            action = 0
            
        #print('Returning action=',action)
        return int(action)

    def animate(i):
        """perform animation step"""

        global x,angle, omega, timeElapsed, currV, oldV, world, Qval

        fail = 0

        if angle > 75*(np.pi/180) and angle < 105*(np.pi/180) and x < 0.5 and x > -0.5:

            currV = policy(world,currV,omega,x,angle)
            x, angle,omega,timeElapsed = model(x,angle,oldV,currV,omega,timeElapsed)

            oldV = currV

        else:
            fail = 1

        line.set_data([x,x+np.cos(angle)],[0,np.sin(angle)])
        time_text.set_text('time = %.1f' % timeElapsed)
        return line, time_text

    # choose the interval based on dt and the time to animate one step

    from time import time
    t0 = time()
    animate(0)
    
    t1 = time()
    interval = dt*1000 - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=1000,
                              interval=interval, blit=True, init_func=init,repeat=True)

    plt.show()

