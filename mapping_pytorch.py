#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import wandb
from omegaconf import OmegaConf as omg
import torch
import numpy as np
from agent import Agent
from trainer import Trainer, reward_fn
from data_generator_v2 import DataGenerator, SimulatedAnnealing
import time
import os
import pandas as pd


# In[2]:



def load_conf():
    """Quick method to load configuration (using OmegaConf). By default,
    configuration is loaded from the default config file (config.yaml).
    Another config file can be specific through command line.
    Also, configuration can be over-written by command line.

    Returns:
        OmegaConf.DictConfig: OmegaConf object representing the configuration.
    """
    default_conf = omg.create({"config" : "config.yaml"})

    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    yaml_file = omg.merge(default_conf, cli_conf).config

    yaml_conf = omg.load(yaml_file)

    return omg.merge(default_conf, yaml_conf, cli_conf)


# In[3]:


def main():
    conf = load_conf()
    wandb.init(project=conf.proj_name, config=dict(conf))

    agent = Agent(space_dim= conf.dimension, embed_hidden=conf.embed_hidden, enc_stacks=conf.enc_stacks, ff_hidden=conf.ff_hidden, enc_heads=conf.enc_heads, query_hidden=conf.query_hidden, att_hidden=conf.att_hidden, crit_hidden=conf.crit_hidden, n_history=conf.n_history, p_dropout=conf.p_dropout)
    wandb.watch(agent)

    dataset = DataGenerator()

    trainer = Trainer(conf, agent, dataset)
    trainer.run()

    # Save trained agent
    dir_ = str(conf.dimension)+'D_'+'MESH'+str(conf.max_len) +'_b'+str(conf.batch_size)+'_e'+str(conf.embed_hidden)+'_n'+str(conf.ff_hidden)+'_s'+str(conf.enc_stacks)+'_h'+str(conf.enc_heads)+ '_q'+str(conf.query_hidden) +'_a'+str(conf.att_hidden)+'_c'+str(conf.crit_hidden)+ '_lr'+str(conf.lr)+'_d'+str(conf.lr_decay_steps)+'_'+str(conf.lr_decay_rate)+ '_steps'+str(conf.steps)
    path = "save/"+dir_
    if not os.path.exists(path):
        os.makedirs(path)
    save_path= str(path)+'/'+str(conf.model_path)
    torch.save(agent.state_dict(), save_path)

    input_test = []
    for _ in range (conf.batch_size): 
        input_test1 = np.loadtxt("mpeg_4x3.txt", delimiter=",")
        input_test.append(input_test1)

    if conf.test:
        device = torch.device(conf.device)
        # Load trained agent
        agent.load_state_dict(torch.load(save_path))
        agent.eval()
        agent = agent.to(device)
        start_time = time.time()
        running_reward = 0
        for _ in range(conf.test_steps):
            #input_batch = dataset.test_batch(conf.batch_size, conf.max_len, conf.dimension, shuffle=False)
            input_batch = torch.Tensor(input_test).to(device)

            tour, *_ = agent(input_batch)

            reward = reward_fn(input_batch, tour, conf.x_dim, conf.batch_size)

            # Find best solution
            j = reward.argmin()
            best_tour = tour[j][:].tolist()

            # Log
            running_reward += reward[j]

            # Display
            print('Reward (before 2 opt)', reward[j],'tour', best_tour)
            opt_tour, opt_length = dataset.loop2opt(input_batch.cpu()[j], best_tour, conf.x_dim)
            print('Reward (with 2 opt)', opt_length, 'opt_tour',opt_tour)
            #dataset.visualize_2D_trip(opt_tour)

        wandb.run.summary["test_reward"] = running_reward / conf.test_steps
        print("--- %s seconds ---" % (time.time() - start_time))


# In[4]:


if __name__ == "__main__":
    main()

