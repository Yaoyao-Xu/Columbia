
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d





rob = [(-2,0),(-2,0),(-1,0),(-2,1),(-1,1)]
obs = [(-1,0),(0,0),(0,1),(1,1)]



def get_cspace_obstacle(rob, obs):
  x_diff = 0
  y_diff = 0
  if rob[0] != (0,0):
    x_diff = rob[0][0] - 0
    y_diff = rob[0][1] - 0
  rob_reflect = [(-(each[0] - x_diff), -(each[1] - y_diff)) for each in rob]
  new_vertices = obs.copy()
  for vertice in obs:
    x_diff = rob_reflect[0][0] - vertice[0]
    y_diff = rob_reflect[0][1] - vertice[1]
    for each in rob_reflect:
      if (each[0]-x_diff, each[1]-y_diff) not in new_vertices:
        new_vertices.append((each[0]-x_diff, each[1]-y_diff))
  return new_vertices
    


# In[4]:


#first vertice and last vertice are the same
points = get_cspace_obstacle(rob,obs)
print(points)


# In[5]:


def cspace_obstacle_convexhull(points):
  points = np.array(points)
  print(points)
  hull = ConvexHull(points)
  plt.plot(points[:,0], points[:,1],'o')
  for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1],'k-')




cspace_obstacle_convexhull(points)


rng = np.random.default_rng()
points = rng.random((30, 2))   # 30 random points in 2-D
print(points)
print(points[:,0])
hull = ConvexHull(points)
print(hull)
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)


