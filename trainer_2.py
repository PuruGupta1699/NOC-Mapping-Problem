import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx



def reward_fn(coords, tour, x_dim, batch_size):
    """Reward function. Compute the total distance for a tour, given the
    coordinates of each city and the tour indexes.

    Args:
        coords (torch.Tensor): Tensor of size [batch_size, seq_len, dim],
            representing each city's coordinates.
        tour (torch.Tensor): Tensor of size [batch_size, seq_len + 1],
            representing the tour's indexes (comes back to the first city).

    Returns:
        float: Reward for this tour.
    """
    reward1= []
    #app_input=pd.DataFrame(inputs)
    app_input=coords.cpu()
    app_input =app_input.numpy()

    for i in range (batch_size):
       #cost1 = cost (tour[i])
       #reward1.append(cost1)
        in_put=tour[i].cpu().numpy()
        #print(in_put)
        #e_w=tf.zeros([1,],tf.int64)
        e_w=0
        #reward1 = tf.zeros([128],tf.int64)
        #print(self.app)
        gg=nx.from_numpy_array(app_input[i])
        app = nx.to_pandas_edgelist(gg)
        #print(app)
        #aaa=pd.DataFrame(app_input[i])
        #app=nx.to_pandas_edgelist(nx.from_pandas_adjacency(aaa))
        app_len= len(app)
        for n in range (app_len):

            a=np.argwhere(in_put==app.source[n])
            b=np.argwhere(in_put==app.target[n])

            x1=a//x_dim
            y1=a%x_dim

            x2=b//x_dim
            y2=b%x_dim
            #print(a,b)
            c=abs(x1-x2)+abs(y1-y2)

            e=c*app.weight[n]

            e_w=e_w+e[0,]
        #elf.cost1=tf.reduce_sum(e_w)
        reward1.append(e_w)
        #print(reward1)
    reward1= torch.reshape(torch.tensor(reward1, dtype=torch.float), (-1,))
    #reward = tf.cast(reward1,tf.float32)
    
    return reward1


class Trainer():
    def __init__(self, conf, agent, dataset):
        """Trainer class, taking care of training the agent.

        Args:
            conf (OmegaConf.DictConf): Configuration.
            agent (torch.nn.Module): Agent network to train.
            dataset (data.DataGenerator): Data generator.
        """
        super().__init__()

        self.conf = conf
        self.agent = agent
        self.dataset = dataset

        self.device = torch.device(self.conf.device)
        self.agent = self.agent.to(self.device)
        params = list(self.agent.embedding.parameters()) + list(self.agent.encoder.parameters()) +list(self.agent.decoder.parameters())
        
        self.optim2 = torch.optim.Adam(params=self.agent.critic.parameters(), lr=self.conf.lr)
        self.optim1 = torch.optim.Adam(params= params, lr=self.conf.lr)
        
        gamma = 1 - self.conf.lr_decay_rate / self.conf.lr_decay_steps      # To have same behavior as Tensorflow implementation
        self.scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optim1, gamma=gamma)
        self.scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optim2, gamma=gamma)


    def train_step(self, data, x_dim, batch_size):
        #self.optim.zero_grad()
        self.optim1.zero_grad()
        # Forward pass
        tour, critique, log_probs, _ = self.agent(data)

        # Compute reward
        reward = reward_fn(data, tour, x_dim, batch_size)
        reward = torch.Tensor(reward).to(self.device)
        # Compute losses for both actor (reinforce) and critic
        loss1 = ((reward - critique).detach() * log_probs).mean()
        loss2 = F.mse_loss(reward, critique)

        # Backward pass
        
        loss1.backward()
        loss2.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.conf.grad_clip)

        # Optimize
        self.optim1.step()
        self.optim2.step()

        # Update LR
        self.scheduler1.step()
        self.scheduler2.step()

        return reward.mean(), [loss1, loss2]

    def run(self):
        self.agent.train()
        running_reward, running_losses = 0, [0, 0]
        for step in range(self.conf.steps):
            input_batch = self.dataset.train_batch(self.conf.batch_size, self.conf.max_len, self.conf.dimension, self.conf.x_dim, self.conf.y_dim)
            input_batch = torch.Tensor(input_batch).to(self.device)

            reward, losses = self.train_step(input_batch, self.conf.x_dim, self.conf.batch_size)

            running_reward += reward
            running_losses[0] += losses[0]
            running_losses[1] += losses[1]

            if step % self.conf.log_interval == 0 and step != 0:
                # Log stuff
                wandb.log({
                    'reward': running_reward / self.conf.log_interval,
                    'actor_loss': running_losses[0] / self.conf.log_interval,
                    'critic_loss': running_losses[1] / self.conf.log_interval,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'step': step
                })

                # Reset running reward/loss
                running_reward, running_losses = 0, [0, 0]


            
