# Title:
# made by:
# import here
from math import sqrt
from queue import PriorityQueue

# read data from txt
f = open("tsp.txt", "r")
data = []
for line in f:
    data.append(line[:-1])

# create test object, which is a 2-D list
# Using tuple makes program run faster
nodes = []
for node in data:
    nodes.append(tuple(node.split(",")))
nodes = tuple(nodes)


# The function used to find short path
# It takes in start point and the created test object
# User can change start point to get different result
def find_short(start, nodeset):
    # This inner function is to find distance between point1 and point2
    def distance(p1, p2):
        return sqrt((float(p1[0]) - float(p2[0])) ** 2 + ((float(p1[1]) - float(p2[1])) ** 2))

    # This inner function is to find all possible connects for one given point, regardless distance
    # So, each point is child of each other
    def successors(parent):
        # result list
        children = []
        # operate in test object
        for n in nodeset:
            # if n is not parent itself, let n be its child
            if parent != n:
                children.append(n)
        return children

    # PriorityQueue that tracks the current point
    pq = PriorityQueue()
    # Always start at start point
    pq.put((0, start))
    # Path list is the result list shows the sequence of points
    path = []
    # Always start at start point
    path.append(start)
    # total distance tracker
    d = []

    # set() that check for re-visit, which is not allowed
    # set() makes program run faster
    visited = set()

    # when pq has point
    while not pq.empty():
        # pop current node
        _, current = pq.get()
        # a helper that used to find closest point for current node
        # 1000 is an ideal value that greater than largest distance in 1000 nodes set
        compareL = 100
        #minode = tuple()
        # # check for re-visit, if current node is connected, skip it
        if current not in visited:
            # get all children of current node
            for child in successors(current):
                # check for re-visit, if child node is connected, skip it
                if child not in visited:
                    # if not, calculate distance
                    length = distance(current, child)
                    # if the distance is less than previous one, replace it
                    # and set min_node equals to this child
                    if length <= compareL:
                        compareL = length
                        minode = child
            # when reach here, a min_node that closest to the current point must be found
            # add it to result list
            path.append(minode)
            # push new found min_node into pq, so it start with min_node in next round
            pq.put((0, minode))
            # finished all work for current node, mark it or save it to visited
            visited.add(current)
    # do some compensate to the last point
    path.pop()
    path.append(start)

    for i in range(1000):
        d.append(distance(path[i], path[i+1]))

    # return
    return path, sum(d)


# try 1000 nodes as start point
fully_check = dict()
for i in range(1000):
    fully_check[i] = find_short(nodes[i], nodes)[1]
# find index of whose outcome is smallest
min_key = min(fully_check, key=lambda k: fully_check[k])

sequence,distance = find_short(nodes[min_key], nodes)
print(distance)
file = open(r'result.txt', mode='w')
for line in sequence:
    file.write(line[0])
    file.write(',')
    file.write(line[1])
    file.write('\n')
file.close()