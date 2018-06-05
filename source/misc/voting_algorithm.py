from collections import defaultdict
import numpy as np
import math
from collections import Counter
import operator as op


M = np.matrix([[25, 30, 10, 15],
               [30, 15, 10, 25],
               [25, 15, 10, 30],
               [15, 25, 30, 10]])

def _get_original_voting(M):
    original_voting = defaultdict(lambda: defaultdict(lambda: int()))
    it = np.nditer(M, flags=['multi_index'])
    while not it.finished:
        original_voting[int(it[0])][it.multi_index[1]] += 1
        it.iternext()    

        
def add_or_sub_init_votes(original_voting, row, voting_power, op):
    for pos, elm in enumerate(row):
        original_voting[elm][pos] = op(original_voting[elm][pos], voting_power)
    

def recast_vote_matrix(M, init_row_voting_power):
    new_M = np.zeros(M.shape, dtype=np.int)
    for i, row in enumerate(M):
        # add inital voting power to the current row
        add_or_sub_init_votes(original_voting, np.array(row)[0], init_row_voting_power, op.add)

        # compute averages and reposition the ranks accordingly
        repositions = []
        for user_id, pos_votes in original_voting.items():
            avg_rank = sum(pos*freq for pos, freq in pos_votes.items()) / len(M)
            repositions.append({'user_id': user_id, 'rank': avg_rank})
        df = pd.DataFrame(repositions)
        df.sort_values(by='rank', inplace=True)
        new_M[i,:] = df['user_id'].values
        
        # remove inital voting power to the current row
        # to obtain the original votes again
        add_or_sub_init_votes(original_voting, np.array(row)[0], init_row_voting_power, op.sub)
    return new_M

recast_vote_matrix(M, 3)

# print(json.dumps(original_voting, indent=4))
