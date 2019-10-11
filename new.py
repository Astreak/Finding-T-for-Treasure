import numpy as np

import time
import pandas as pd
np.random.seed(2)
ACTIONS=["left","right"]
EPSILON=0.9
ALPHA=0.1
GAMMA=0.9
N_STATE=6
EPISODES=13
FRESH_TIME=0.3
a=np.zeros((N_STATE,len(ACTIONS)))


def c_q_table():
    data=pd.DataFrame(a,columns=ACTIONS)
    return data

def choose(state,data):
    state_action=data.iloc[state,:]
    if (np.random.uniform()>EPSILON or state_action.all()==0):
        action_state=np.random.choice(ACTIONS)
    else:
        action_state=state_action.argmax()
    return action_state


def get_env(S,A):
    if A=="right":
        if S==N_STATE-2:
            R=1
            S_="terminal"
        else:
            S_=S+1
            R=0
            
        
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R


def get_update(S,episode,step_count):
    env_S=["-"]*(N_STATE-1)+["T"]
    if S=="terminal":
        interaction="EPISODE:%s ,Tsteps:%s"%(episode+1,step_count)
        print("\r{}".format(interaction),end="")
        time.sleep(2)
        print("\r                        ",end="")
    else:
        env_S[S]="o"
        interaction="".join(env_S)
        print("\r{}".format(interaction),end="")
        time.sleep(FRESH_TIME)

def rl():
    q_table=c_q_table()
    
    for episode in range(EPISODES):
        S=0
        
        step_count=0
        terminate=False
        get_update(S,episode,step_count)
        while not terminate:
            A=choose(S,q_table)
            S_,R=get_env(S,A)
            q_predict=q_table.ix[S,A]
            if S_!="terminal":
                q_target=R+GAMMA*q_table.iloc[S_,:].max()
            else:
                q_target=R
                terminate=True
            q_table.ix[S,A]+=ALPHA*(q_target-q_predict)
            S=S_
            get_update(S,episode,step_count+1)
            step_count+=1
    return q_table


if __name__=="__main__":
    q_table=rl()
    print("\r\nQ-TABEL:\n")
    print(q_table)
    
                
            
            
    
    
            

            


