def closest_euclidean(q, qp):
    """
    :param q, qp: Two 2D vectors in S1 x S1
    :return: qpp, dist. qpp is transformed version of qp so that L1 Euclidean distance between q and qpp
    is equal to toroidal distance between q and qp. dist is the corresponding distance.
    """
    q = np.array(q)
    qp = np.array(qp)

    A = np.meshgrid([-1,0,1], [-1,0,1])
    qpp_set = qp + 2*np.pi*np.array(A).T.reshape(-1,2)
    distances = np.linalg.norm(qpp_set-q, 1, axis=1)
    ind = np.argmin(distances)
    dist = np.min(distances)

    return qpp_set[ind], dist

def clear_path(arm, q1, q2):
    """
    :param arm: NLinkArm object
    :param q1, q2: Two configurations in S1 x S1
    :return: True if edge between q1 and q2 sampled at EDGE_INC increments collides with obstacles, False otherwise
    """
    q2p, dist = closest_euclidean(q1,q2)  #from q1 to q1p(q2prime) is the shorter segment 
    q_angle = np.arctan2(q2p[1]-q1[1], q2p[0]-q1[0])
    #print(f"q2p is {q2p}, dist is {dist}, q_angle is {q_angle}")
    
    #continue add EDGE_INC
    config = np.array(q1)
    while (EDGE_INC < dist) :
        config[0] += np.cos(q_angle) * EDGE_INC
        config[1] += np.sin(q_angle) * EDGE_INC
        if detect_collision(arm, config) == True:
            return True
        else:
            dist -= EDGE_INC
    if detect_collision(arm, q2p) == True:
        return True
    else:
        return False

def find_qnew(arm, tree, qrand):
    """
    :param tree: RRT dictionary {(node_x, node_y): (parent_x, parent_y)}
    :param qrand: Randomly sampled configuration
    :return: qnear in tree, qnew between qnear and qrand with distance DELTA from qnear
    """
    #dists includes (dist, key)
    dists = [(closest_euclidean(qrand, key)[1], key) for key in tree.keys()]
    qnear = sorted(dists, key = lambda x: x[0])[0][1]
    qrandp, dist = closest_euclidean(qnear, qrand)
    qnew = [0,0]
    if dist < DELTA:
        qnew = qrandp
    else:
        q_angle = np.arctan2(qrandp[1]-qnear[1], qrandp[0]-qnear[0])
        qnew[0] = qnear[0] + np.cos(q_angle) * DELTA
        qnew[1] = qnear[1] + np.sin(q_angle) * DELTA
    
    if clear_path(arm, qnear, qnew):
        qnew = None
    else:
        if qnew[0] > np.pi:
            qnew[0] -= 2*np.pi
        elif qnew[0] < -np.pi:
            qnew[0] += 2*np.pi 
        if qnew[1] > np.pi:
            qnew[1] -= 2*np.pi
        elif qnew[1] < -np.pi:
            qnew[1] += 2*np.pi
        qnew = tuple(qnew)
    
    return qnear, qnew

def find_qnew_greedy(arm, tree, qrand):
    """
    :param arm: NLinkArm object
    :param tree: RRT dictionary {(node_x, node_y): (parent_x, parent_y)}
    :param qrand: Randomly sampled configuration
    :return: qnear in tree, qnew between qnear and qrand as close as possible to qrand in increments of DELTA
    """
    #dists includes (dist, key)
    qnear, qnew = find_qnew(arm, tree, qrand)
    final_qnew = [0,0] #final_qnew is used to store the latest qnew which will be returned
    if qnew == None:
        final_qnew = None
        return qnear, final_qnew
    else:
        qrandp, dist = closest_euclidean(qnew, qrand)
        q_angle = np.arctan2(qrandp[1]-qnew[1], qrandp[0]-qnew[0])
        while True:
            new_qnew = [0,0] #new_qnew is the next qnew after the current qnew, clear_path will be used afterwards
            if dist < DELTA:
                new_qnew = qrandp
                if clear_path(arm, qnew, new_qnew):
                    final_qnew = qnew
                    break
                else:
                    final_qnew = qrandp
                    break
            else:  
                new_qnew[0] = qnew[0] + np.cos(q_angle) * DELTA
                new_qnew[1] = qnew[1] + np.sin(q_angle) * DELTA
            if clear_path(arm, qnew, new_qnew):
                final_qnew = qnew
                break
            else:
                qnew = new_qnew
                dist -= DELTA
    if final_qnew[0] > np.pi:
        final_qnew[0] -= 2*np.pi
    elif final_qnew[0] < -np.pi:
        final_qnew[0] += 2*np.pi 
    if final_qnew[1] > np.pi:
        final_qnew[1] -= 2*np.pi
    elif final_qnew[1] < -np.pi:
        final_qnew[1] += 2*np.pi
    final_qnew = tuple(final_qnew)
        
        
    return qnear, final_qnew

def construct_tree(arm):
    """
    :param arm: NLinkArm object
    :return: roadmap: Dictionary of nodes in the constructed tree {(node_x, node_y): (parent_x, parent_y)}
    :return: path: List of configurations traversed from start to goal
    """
    reach_goal = False
    tree = {START:None}
    for i in range(1,MAX_NODES):
        np.random.seed()
        if np.random.rand() > BIAS:
            qrand = (np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi))
        else:
            qrand = GOAL
        #change the function between(find_qnew, find_qnew_greedy) to see the different strategies
        #qnear, qnew = find_qnew_greedy(arm, tree, qrand)
        qnear, qnew = find_qnew_greedy(arm, tree, qrand)
        if qnew == None:
            continue
        _,dist2goal = closest_euclidean(qnew, GOAL)
        if dist2goal <= DELTA:
            #set qnew = GOAL
            if clear_path(arm, qnear, GOAL):
                continue
            else:
                tree[GOAL] = qnear
                reach_goal = True
                break
        else:
            tree[qnew] = qnear
            
    if reach_goal == True:
        route = [GOAL]
        tree_temp = tree.copy()
        while True:
            parent = tree_temp.pop(route[-1])
            route.append(parent) 
            if parent == START:
                break
        route = route[::-1]
        return tree, route
    else:
        return tree,[]