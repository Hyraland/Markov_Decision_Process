import numpy as np 
import rl_gridworld as gw

delta = 1e-3

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


if __name__ == '__main__':
    g = gw.windy_grid()
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

    V = {}
    for s in g.all_states():
        V[s] = 0

    print_policy(policy_prob, g)
    print_value(V, g)

    gamma = 0.9

    it = 0
    while True:
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
    print('After iteration:')
    print_policy(policy_prob, g)
    print_value(V, g)











