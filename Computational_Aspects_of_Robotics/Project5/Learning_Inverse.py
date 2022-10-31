import random
Input = []
Output = []
for _ in range(1000):
  q1 = random.uniform(-np.pi, np.pi)
  q2 = random.uniform(-np.pi, np.pi)
  q3 = random.uniform(-np.pi, np.pi)
  X=cos(q1)+cos(q1)*cos(q2)+cos(q1)*cos(q2)*cos(q3)-cos(q1)*sin(q2)*sin(q3)
  Y=sin(q1)+cos(q2)*sin(q1)-sin(q1)*sin(q2)*sin(q3)+cos(q2)*cos(q3)*sin(q1)
  Z=sin(q2)+cos(q2)*sin(q3)+cos(q3)*sin(q2)

  Output.append([q1, q2, q3])
  Input.append([X, Y, Z])

Input = np.array(Input)
Output = np.array(Output)
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation="relu"),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3)
])



model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(Input, Output, validation_split=0.2, epochs=60, batch_size=32, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
K = 500
traj = np.zeros((K,3))
traj[:,0] = 2*np.cos(np.linspace(0,2*np.pi,num=K))
traj[:,1] = 2*np.sin(np.linspace(0,2*np.pi,num=K))
traj[:,2] = np.sin(np.linspace(0,8*np.pi,num=K))

prediction = model.predict(traj)

fig = plt.figure()
ax = plt.axes(projection ='3d')
z = np.sin(np.linspace(0,8*np.pi,num=K))
x = 2*np.cos(np.linspace(0,2*np.pi,num=K))
y = 2*np.sin(np.linspace(0,2*np.pi,num=K))
 
# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('The End Effector trajectory')
plt.show()

fig = plt.figure()
ax = plt.axes(projection ='3d')
z = prediction[:,2]
x = prediction[:,0]
y = prediction[:,1]
 
# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('The End Effector trajectory')
plt.show()