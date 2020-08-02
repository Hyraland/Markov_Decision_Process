import numpy as np 
import rl_gridworld as gw

def print_policy(P, g):
    r, c = g.rows, g.columns
    Ps = [[{} for j in range(c)] for i in range(r)]
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

def policy_evaluation(policy_prob, g, gamma, delta):
    V = {}
    for s in g.all_states():
        V[s] = 0

    it = 0
    while True:
        it += 1
        itermax = 0
        V_ = {}
        for s in g.all_states():
            actions = g.actions.get(s, ())
            V_[s] = 0
            for a in actions:
                probs = g.probs.get((s,a), {})
                for s_ in list(probs.keys()):
                    r = g.rewards.get(s_, 0)
                    V_[s] += policy_prob.get(s, {}).get(a, 0)*probs[s_]*(r + gamma*V[s_])
            itermax = max(itermax, abs(V_[s]-V[s]))
        if itermax < delta: break
        V = V_
    return V

def policy_iteration(policy_prob, V, g):
    old_pp = policy_prob
    while True:
        policychange = False
        new_pp = {}
        for s in g.all_states():
            qmax, amax = 0, 0
            actions = g.actions.get(s, ())
            for a in actions:
                qn = float('-inf')
                probs = g.probs.get((s,a), {})
                for s_ in list(probs.keys()):
                    r = g.rewards.get(s_, 0)
                    qn += probs[s_]*(r + gamma*V[s_])
                if qn >= qmax:
                    qmax = qn
                    amax = a
            new_pp[s] = {a: 1.0}
            if new_pp[s] != old_pp.get(s, {}): 
                policychange = True
        if policychange == False: break
        old_pp = new_pp

    return policy_prob




if __name__ == '__main__':
    g = gw.windy_grid()
    g = gw.windy_grid_penalized(g, -0.1)
    # 1(0001) for Right, 2(0010) for Left, 4(0100) for Down, 8(1000) for Up
    policy_prob = {
    (2, 0): {8: 0.5, 1: 0.5},
    (1, 0): {8: 1.0},
    (0, 0): {1: 1.0},
    (0, 1): {1: 1.0},
    (0, 2): {1: 1.0},
    (1, 2): {8: 1.0},
    (2, 1): {1: 1.0},
    (2, 2): {8: 1.0},
    (2, 3): {2: 1.0},
    }
    delta = 1e-3
    gamma = 0.9

    V = policy_evaluation(policy_prob, g, gamma, delta)
    P = policy_iteration(policy_prob, V, g)

    print('After iteration:')
    print_policy(P, g)
    print_value(V, g)
