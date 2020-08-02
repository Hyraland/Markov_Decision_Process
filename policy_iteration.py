import numpy as np 
import gridworld as gw

def print_policy(P, g):
    r, c = g.rows, g.columns
    Ps = [[0 for j in range(c)] for i in range(r)]
    poses = list(P.keys())
    for pos in poses:
        Ps[pos[0]][pos[1]] = P.get(pos)
    print(Ps)



def print_value(V, g):
    r, c = g.rows, g.columns
    Vs = [[0 for j in range(c)] for i in range(r)]
    poses = list(V.keys())
    for pos in poses:
        Vs[pos[0]][pos[1]] = V.get(pos)
    print(Vs)

def policy_evaluation(policy, g, gamma, delta):
    V = {}
    for s in g.all_states():
        V[s] = 0

    it = 0
    while True:
        it += 1
        itermax = 0
        V_ = {}
        for s in g.all_states():
            a = policy.get(s, 0)
            V_[s] = 0
            probs = g.probs.get((s,a), {})
            for s_ in list(probs.keys()):
                r = g.rewards.get(s_, 0)
                V_[s] += probs[s_]*(r + gamma*V[s_])
            itermax = max(itermax, abs(V_[s]-V[s]))
        if itermax < delta: break
        V = V_
    return V

def policy_iteration(policy, g, gamma, delta):
    old_pp = policy
    while True:
        V = policy_evaluation(old_pp, g, gamma, delta)
        policychange = False
        new_pp = {}
        for s in g.all_states():
            qmax, amax = float('-inf'), 0
            old_a = old_pp.get(s, 0)
            for a in g.actions.get(s, ()):
                qn = 0
                probs = g.probs.get((s,a), {})
                for s_ in list(probs.keys()):
                    r = g.rewards.get(s_, 0)
                    qn += probs[s_]*(r + gamma*V[s_])
                if qn >= qmax:
                    qmax = qn
                    amax = a
            new_pp[s] = amax
            if new_pp[s] != old_pp.get(s, 0): 
                policychange = True
        if policychange == False: break
        old_pp = new_pp
    return new_pp

def policy_initialization(g):
    policy = {}
    for s in g.actions.keys():
        policy[s] = np.random.choice(g.actions[s])
    return policy


if __name__ == '__main__':
    g = gw.windy_grid()
    g = gw.windy_grid_penalized(g, -0.4)
    # 1(0001) for Right, 2(0010) for Left, 4(0100) for Down, 8(1000) for Up
    delta = 1e-3
    gamma = 0.9

    policy = policy_initialization(g)
    print('Initial:')
    print_policy(policy, g)
    print_value(g.rewards, g)

    P = policy_iteration(policy, g, gamma, delta)
    V = policy_evaluation(P, g, gamma, delta)

    print('After iteration:')
    print_policy(P, g)
    print_value(V, g)
