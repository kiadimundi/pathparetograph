from rdp import rdp
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing


# p1=0
# p2=1
# realPath=np.column_stack((paths2[p1][:,0],paths2[p1][:,1]))
# pa1=rdp(realPath,epsilon)
# # pa2=rdp(np.column_stack((paths2[p2][:,0],paths2[p2][:,1])),epsilon)
# # distanceDf=similaritymeasures.frechet_dist(pa1, pa2)  
# print(len(realPath),len(pa1))
# for p1 in tqdm(range(len(paths2))):
#     for p2 in range(p1+1,len(paths2)):
#         pa1=rdp(np.column_stack((paths2[p1][:,0],paths2[p1][:,1])),epsilon)
#         pa2=rdp(np.column_stack((paths2[p2][:,0],paths2[p2][:,1])),epsilon)
#         distanceDf=similaritymeasures.frechet_dist(pa1, pa2)
#         distance_matrix[p1,p2]=distanceDf

# rdppaths = {}

def helper(args):
    return simplifyPathFromSet(*args)

def simplifyPathFromSet(p1,paths2,epsilon=0.005):
    # for p1 in tqdm(range(len(paths2))):
    fullPath=np.column_stack((paths2[p1][:,0],paths2[p1][:,1]))
    pa1=simplifyPath(fullPath,epsilon)
    
    # until=len(paths2)
    # for p2 in range(p1+1,until):
    #     time.sleep(0.1)
        # pa1=rdp(np.column_stack((paths2[p1][:,0],paths2[p1][:,1])),epsilon)
        # pa2=rdp(np.column_stack((paths2[p2][:,0],paths2[p2][:,1])),epsilon)
        # distanceDf=similaritymeasures.frechet_dist(pa1, pa2)
        # distance_matrix[p1,p2]=distanceDf
    return pa1

def simplifyPath(path,epsilon=0.005):
    simplePath = rdp(path,epsilon)
    return simplifyPath

def loadDataAndSimplifyPaths():
    with open('data.pickle', 'rb') as f:
        commons = pickle.load(f)
        commonsNumber= pickle.load(f)
        paths2 = pickle.load(f)
        unObj = pickle.load(f)
        df = pickle.load(f)

    # distance_matrix=np.zeros((len(paths2),len(paths2)))
    epsilon=0.005
    numbers = range(len(paths2))
    # numbers = range(2)
    with multiprocessing.Pool(processes=7) as pool:
        # args = [(number, paths2, epsilon) for number in numbers]
        args = zip(numbers, [paths2]*len(paths2),[epsilon]*len(paths2))
        # results = list(tqdm(executor.map(task_function, numbers), total=len(numbers)))
        results = list(tqdm(pool.imap(helper, args), total=len(numbers)))
        
    # for i in numbers:
    #     rdppaths[p1]=results[i]
    with open('rdpp.pickle', 'wb') as f:
        pickle.dump(results, f)
    return results

def loadSimplifiedPathsFromFile(file='rdpp.pickle'):
    with open(file, 'rb') as f:
        paths = pickle.load(f)
    return paths

# if __name__ == '__main__':
#     res=main()
    # print(res)
    
    