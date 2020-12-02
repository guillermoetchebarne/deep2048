import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import load_model
from keras.optimizers import Adam
from logic import game_2048
from exp_replay import ExperienceReplay
from models import create_mlp
from time import time

# parameters
grid_shape = (4, 4)
epochs = 10000
epsilon = np.linspace(0.01, 0.99, num = epochs//2)  # 
epsilon = np.flip(epsilon)
epsilon = np.append(epsilon, 0.025*np.ones(epochs//2))
num_actions = 4  
max_memory = 8192
batch_size = 128
input_size = grid_shape[0]*grid_shape[1]
lr = .0001

# optimizer
opt = Adam(lr = lr, decay = 1e-3/200)

# model
model = create_mlp(input_size = input_size, layers = [128, 256, 512, 256, 128], output_size = 4)
model.compile(optimizer = opt, loss = "mse")
model.summary()

# define environment
env = game_2048(grid_shape = grid_shape)

# initialize experience replay object
exp_replay = ExperienceReplay(max_memory = max_memory, env_dim = input_size,
							  discount = .99)

# training loop
losses = []
ti = time()
for e in range(epochs):
    game_over = False
    env.reset()
    n_steps = 0
    while game_over == False:
        n_steps += 1
        old_state = env.get_state()
        legal_moves = list(env.next_states.keys())
        if np.random.rand() <= epsilon[e]:
            a = int(np.random.choice(legal_moves))
        else:
            q = model.predict(old_state)[0]
            aux = np.argmax(q[legal_moves])
            a = legal_moves[aux]

        new_state, reward, game_over = env.update(a)
        # remember experience
        exp_replay.remember(old_state, a, new_state, reward, game_over)
        
    # adapt model
    inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
    loss = model.train_on_batch(inputs, targets)
    losses.append(loss)

    print("Epoch {:03d}/{:03d} | Loss {:.5f} | Steps {} | Max tile: {} | Epsilon {:.3f}".format(e, epochs, 
                losses[-1], n_steps, round(2**(np.max(env.get_state())*10)),epsilon[e]))
    
print("Done training. Time elapsed: {:.2f} hours.".format((time() - ti)/3600))
print("Saving model...")
model.save("model")
print("Saved.")
np.savetxt("losses.txt", losses)
