import random
import math
import numpy as np
from operator import add
from cvxopt import matrix, solvers
solvers.options['show_progress'] = 0

class BlottoLP:
    def __init__(self, n_sol_atk, n_sol_def, n_bfs):
        self.n_sol_attacker = n_sol_atk
        self.n_sol_defender = n_sol_def
        self.n_battlefields = n_bfs
        self.strat_space_A = None
        self.strat_space_D = None

    def unique_depl_strats(self, depl_strat_list):
        '''Extract and return unique deplyment strategies'''
        strats = [s for s in depl_strat_list]

        for idx, s in enumerate(strats):
            kidx = idx + 1
            while kidx < len(strats):
                if set(s) == set(strats[kidx]):
                    del strats[kidx]
                else:
                    kidx += 1
        
        return strats

    def list_strat(self, n_sol, n_bfs):
        '''
        list_strat creates a matrix with all strategies for
        a number of armies

        Lists all the strategies that a player has where
          no of troops = n_sol
          no of bases = n_bfs

        The strategies are given in a matrix where each row
        represents a strategy
        '''

        strategies = []

        if n_bfs == 2:
            i = n_sol
            j = 0
            strategies.extend([[i,j]])

            while i > (j + 1):
                i -= 1
                j += 1
                strategies.extend([[i,j]])
        else:
            i = n_sol
            j = 0
            strategies.extend([[i, j] + [0] * (n_bfs - 2)])

            while i > math.ceil(n_sol / n_bfs):
                i -= 1
                j += 1
                strat = self.list_strat(j, n_bfs - 1)
                n_strat = len(strat)
                leading_nums = [[0] for n in range(n_strat)]

                for r in range(n_strat):
                    leading_nums[r] = [i]

                unsorted_strats = map(add, leading_nums, strat)
                sorted_strats = sorted(unsorted_strats, key=lambda s: s[1], reverse=True)

                strategies.extend(sorted_strats)

                strategies = self.unique_depl_strats(strategies)
                
        return strategies

    def game_matrix(self):
        '''
        Creates a matrix for the Blotto Game with
           A = no of troops with the attacker
           D = no of troops with the defender
           B = no of battlefields
        '''

        A = self.n_sol_attacker
        D = self.n_sol_defender
        B = self.n_battlefields

        # Getting the troop deployment strategies
        self.strat_space_A = self.list_strat(A, B)
        n_A_strats = len(self.strat_space_A)

        #print(A_strats)

        self.strat_space_D = self.list_strat(D, B)
        n_D_strats = len(self.strat_space_D)

        #print(D_strats)

        # Zero Initialise the Game Matrix
        gm_mtx = np.zeros([n_A_strats, n_D_strats])

        for a_s_idx in range(n_A_strats):
            for d_s_idx in range(n_D_strats):
                # Reset and initialise the bases captured
                bfs_win = 0

                for a_b in range(B):
                    for d_b in range(B):
                        if self.strat_space_A[a_s_idx][a_b] > self.strat_space_D[d_s_idx][d_b]:
                            bfs_win += 1

                gm_mtx[a_s_idx][d_s_idx] = bfs_win / B
        
        return np.array(gm_mtx)

    def lp_opt_sol(self, gm_mtx, solver="glpk"):
        '''
        Solves Linear Programs in the following form:

        min_{x} f.T @ x
        s.t     A @ x <= b
                A_eq @ x = b_eq
                lb <= x

        This implies that for attacker:
        x.T = [p1, p2, p3, ...., pm, v]
        f.T = [0, 0, 0, ..., -1]
        A = [[-g_{1,1}, ....., -g_{m,1}, 1]
             [-g_{1,2}, ....., -g_{m,2}, 1]
             ....
             [-g_{1,n}, ....., -g_{m,n}, 1]]
        A_eq = [1,1,....,1,0]
        b_eq = 1
        b.T = [0,0,...., 0]

        This implies that for defender:
        x.T = [q1, q2, q3, ...., qn, w]
        f.T = [0, 0, 0, ..., 1]
        A = [[g_{1,1}, ....., g_{m,1}, -1]
             [g_{1,2}, ....., g_{m,2}, -1]
             ....
             [g_{1,n}, ....., g_{m,n}, -1]]
        A_eq = [1,1,....,1,0]
        b_eq = 1
        b.T = [0,0,...., 0]
        '''

        m_mtx, n_mtx = gm_mtx.shape

        '''Solving for Attacker'''
        # f.T denoted as f
        f_A = [0 for i in range(m_mtx)] + [-1]
        f_A = np.array(f_A, dtype="float")
        f_A = matrix(f_A)

        # constraints A @ x <= b
        A = np.matrix(gm_mtx, dtype="float") # reformat each variable is in a row
        A = np.transpose(A)
        A *= -1 # minimization constraint
        new_col = np.ones((n_mtx,1))
        A = np.hstack([A, new_col]) # insert utility column
        A = np.vstack([A, np.eye(m_mtx + 1) * -1]) # > 0 constraint for all vars
        #new_col = np.ones((n_mtx,1))
        #A = np.hstack([A, new_col]) # insert utility column
        A = matrix(A)

        b_A = [0 for i in range(n_mtx)] + [0 for i in range(m_mtx)] + [np.inf]
        b_A = np.array(b_A, dtype="float")
        b_A = matrix(b_A)

        # contraints A_eq @ x = b_eq
        A_eq = [1 for i in range(m_mtx)] + [0]
        A_eq = np.matrix(A_eq, dtype="float")
        A_eq = matrix(A_eq)
        b_A_eq = np.matrix(1, dtype="float")
        b_A_eq = matrix(b_A_eq)

        # solve the LP for Attacker
        sol_A = solvers.lp(c=f_A, G=A, h=b_A, A=A_eq, b=b_A_eq, solver=solver)

        '''Solving for defender'''
        # f.T denoted as f
        f_D = [0 for i in range(n_mtx)] + [1]
        f_D = np.array(f_D, dtype="float")
        f_D = matrix(f_D)

        # constraints D @ x <= b
        D = np.matrix(gm_mtx, dtype="float") # reformat each variable is in a row
        new_col = np.ones((m_mtx,1)) * -1
        D = np.hstack([D, new_col]) # insert utility column
        D = np.vstack([D, np.eye(n_mtx + 1) * -1]) # > 0 constraint for all vars
        D = matrix(D)

        b_D = [0 for i in range(m_mtx)] + [0 for i in range(n_mtx)] + [np.inf]
        b_D = np.array(b_D, dtype="float")
        b_D = matrix(b_D)

        # contraints A_eq @ x = b_eq
        D_eq = [1 for i in range(n_mtx)] + [0]
        D_eq = np.matrix(D_eq, dtype="float")
        D_eq = matrix(D_eq)
        b_D_eq = np.matrix(1, dtype="float")
        b_D_eq = matrix(b_D_eq)

        # solve the LP for Attacker
        sol_D = solvers.lp(c=f_D, G=D, h=b_D, A=D_eq, b=b_D_eq, solver=solver)

        return sol_A, sol_D

    def get_best_strats(self, strat_probs, n, plr_type='attacker'):
        strat_probs = np.array(strat_probs).flatten()[:-1]
        #print(strat_probs)
        sorted_idx = np.argsort(strat_probs)
        # Sorted in descending order
        sorted_idx = sorted_idx[::-1]
        #print(sorted_idx)

        if n < len(sorted_idx):
            best_n_idx = sorted_idx[:n]
        else:
            best_n_idx = sorted_idx[:n]
        #print(best_n_idx)

        if plr_type == 'attacker':
            best_n_strat = np.array(self.strat_space_A)
        else:
            best_n_strat = np.array(self.strat_space_D)

        best_n_strat = best_n_strat[best_n_idx]

        return best_n_strat

    def disp_best_n_strats(self, best_n_strats):
        headers = ["Battlefield " + str(i) for i in range(1, self.n_battlefields + 1)]

        print("\n\n======================================================================================")
        print("Best Troop Deployment Strategies for {} Battlefields with {} Attackers and {} Defenders".format(self.n_battlefields, self.n_sol_attacker, self.n_sol_defender))
        print("======================================================================================\n\n")

        row_format_header = "{:>15}|" * (self.n_battlefields + 1)
        row_format_u = "{:>15}+" * (self.n_battlefields + 1)
        row_format_data = "{:>15}|" + "{:>15}|" * self.n_battlefields
        print(row_format_header.format("", *headers))
        print(row_format_u.format(*(["-"*15]*(self.n_battlefields + 1))))

        for idx, strat in enumerate(best_n_strats):
            print(row_format_data.format("Strategy " + str(idx + 1), *strat))
            print(row_format_u.format(*(["-"*15]*(self.n_battlefields + 1))))

    def save_bestNstrats2csv(self, best_n_strats, plr_type='attacker'):
        headers = ["Battlefield " + str(i) for i in range(1, self.n_battlefields + 1)]

        if plr_type == 'attacker':
            out_report_fname = 'best_attacker_strats_bfs_' + str(self.n_battlefields) + '_atk_' + str(self.n_sol_attacker) + '_def_' + str(self.n_sol_defender) + '.csv'
        else:
            out_report_fname = 'best_defender_strats_bfs_' + str(self.n_battlefields) + '_atk_' + str(self.n_sol_attacker) + '_def_' + str(self.n_sol_defender) + '.csv'

        print("Writing to {}".format(out_report_fname))
        with open(out_report_fname, 'w') as out_f:
            out_f.write("{}\n".format(','.join([""] + headers)))
            for idx, strat in enumerate(best_n_strats):
                row = [str(i) for i in strat]
                out_f.write("{}\n".format(','.join(["Strategy " + str(idx + 1)] + row)))

    def get_payoff(self, lp_soln_A, lp_soln_D):
        payoff = None

        payoff_A = np.array(lp_soln_A).flatten()[-1]
        payoff_D = np.array(lp_soln_D).flatten()[-1]

        if payoff_A == payoff_D:
            payoff = payoff_A
        elif payoff_A > payoff_D:
            payoff = payoff_A
        else:
            payoff = payoff_D

        return payoff

class BlottoPayoffTable:
    def __init__(self):
        self.min_n_sol_A = 1 
        self.min_n_sol_D = 1 
        self.min_n_bfs = 2 
        self.max_n_sol_A = 30 
        self.max_n_sol_D = 30 
        self.max_n_bfs = 9

    def gen_blotto_table(self):
        '''
        Create a 4-D array storing all the information
        for the Colonel Blotto Game

        For parameters:
        A = number of Attacking Armies
        D = number of Defending Armies
        B = number of Bases

        The information stored in the 
        Blotto Table is as follows:
        Blotto Table{A,D,B} = The payoff of the game

        '''
        bfs_mem_size = self.max_n_bfs + 1 - self.min_n_bfs
        n_A_mem_size = self.max_n_sol_A + 1 - self.min_n_sol_A
        n_D_mem_size = self.max_n_sol_D + 1 - self.min_n_sol_D

        # Create Blotto Payoff Table for the ranges above
        blotto_tbl = np.zeros([bfs_mem_size, n_A_mem_size, n_D_mem_size])

        print(blotto_tbl.shape)

        for bfs in range(0, bfs_mem_size):
            n_bfs = bfs + self.min_n_bfs
            print("Generating Blotto Payoff Table for {} bases, {} attackers, {} defenders.....".format(n_bfs, self.max_n_sol_A, self.max_n_sol_D))

            for n_atk in range(0, n_A_mem_size):
                for n_def in range(0, n_D_mem_size):
                    game = BlottoLP(n_atk + 1, n_def + 1, n_bfs)
                    gm_mtx = game.game_matrix()
                    opt_A, opt_D = game.lp_opt_sol(gm_mtx)
                    #blotto_tbl[n_atk, n_def, bfs, 0] = game.get_best_strats(opt_A['x'], 1, plr_type='attacker')
                    #blotto_tbl[n_atk, n_def, bfs, 1] = game.get_best_strats(opt_D['x'], 1, plr_type='defender')
                    blotto_tbl[bfs, n_atk, n_def] = game.get_payoff(opt_A['x'], opt_D['x'])

            print("Blotto Payoff Table Populated for {} bases, {} attackers, {} defenders".format(n_bfs, self.max_n_sol_A, self.max_n_sol_D))

        return blotto_tbl

    def disp_payoff_table(self, payoff_mtx):
        A_idx = [i for i in range(self.min_n_sol_A, self.max_n_sol_A + 1)]
        D_idx = [i for i in range(self.min_n_sol_D, self.max_n_sol_D + 1)]

        for bfs in range(self.max_n_bfs + 1 - self.min_n_bfs):
            print("\n\n=================================")
            print("Blotto Table for Battlefields = {}".format(bfs + self.min_n_bfs))
            print("=================================\n\n")
            payoff_tbl = payoff_mtx[bfs,:,:]

            row_format_header = "{:>10}|" * (self.max_n_sol_D + 1)
            row_format_u = "{:>10}+" * (self.max_n_sol_D + 1)
            row_format_data = "{:>10}|" + "{:>10.6}|" * (self.max_n_sol_D)
            print(row_format_header.format("A/D", *D_idx))
            print(row_format_u.format(*(["-"*10]*(self.max_n_sol_D + 1))))
            for a, row in zip(A_idx, payoff_tbl):
                print(row_format_data.format(a, *row))
                print(row_format_u.format(*(["-"*10]*(self.max_n_sol_D + 1))))

    def save_mtx2csv(self, payoff_mtx):
        A_idx = [str(i) for i in range(self.min_n_sol_A, self.max_n_sol_A + 1)]
        D_idx = [str(i) for i in range(self.min_n_sol_D, self.max_n_sol_D + 1)]

        for bfs in range(self.max_n_bfs + 1 - self.min_n_bfs):
            out_report_fname = 'blotto_payoff_matrix_bfs_' + str(bfs + self.min_n_bfs) + '.csv'
            payoff_tbl = payoff_mtx[bfs,:,:]

            print("Writing to {}".format(out_report_fname))
            with open(out_report_fname, 'w') as out_f:
                out_f.write("{}\n".format(','.join(["A/D"] + D_idx)))
                for a, row in zip(A_idx, payoff_tbl):
                    row = [str(i) for i in row]
                    out_f.write("{}\n".format(','.join([a] + row)))


game_lp = BlottoLP(100,100,3)

gm_mtx = game_lp.game_matrix()
#print(gm_mtx)

opt_A, opt_D = game_lp.lp_opt_sol(gm_mtx)
# print('Attacker Optimal')
# print(np.array(opt_A['x']).shape)
# print(np.array(opt_A['x']).flatten().shape)
# print(np.array(opt_A['x']).flatten()[:-1])
# print('Defender Optimal')
# print(opt_D['x'])

#print('Best 30 Strategies for Attacker')
#best_strats_A = game_lp.get_best_strats(opt_A['x'], 30, plr_type='attacker')
#game_lp.disp_best_n_strats(best_strats_A)
#game_lp.save_bestNstrats2csv(best_strats_A)
# print(best_strats_A)

print('Best 30 Strategies for Defender')
best_strats_D = game_lp.get_best_strats(opt_D['x'], 30, plr_type='defender')
game_lp.save_bestNstrats2csv(best_strats_D, plr_type='defender')
# print(best_strats_D)

# blotto_tbl = BlottoPayoffTable()
# payoff_mtx = blotto_tbl.gen_blotto_table()
#blotto_tbl.disp_payoff_table(payoff_mtx)
# blotto_tbl.save_mtx2csv(payoff_mtx)



