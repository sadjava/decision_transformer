import os
import sys
import random
import wandb
from datetime import datetime
import pyrallis
from dataclasses import dataclass, asdict
from tqdm import tqdm

import numpy as np
import gym

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import D4RLDataset, evaluate_env, get_d4rl_normalized_score, set_seed
from model import DecisionTransformer

@dataclass
class TrainConfig:

    project: str = 'decision_transformer'
    group: str = 'halfcheetah'
    name: str = 'baseline'

    dataset: str = 'medium'
    rtg_scale: float = 1000
    env: str = 'halfcheetah'
    dataset_dir: str = 'd4rl_data'

    context_len: int = 20
    n_blocks: int = 3
    n_heads: int = 1
    embed_dim: int = 128
    dropout_p: float = 0.1

    max_episode_len: int = 1000
    n_episodes: int = 10

    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 10000
    grad_clip: float = 0.25

    max_train_iters: int = 200
    num_updates_per_iter: int = 100

    model_save_dir: str ='models'
    
    device: str = 'cuda'

    train_seed: int = 10
    eval_seed: int = 42
    eval_interval: int = 10


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed)

    if config.env == 'walker2d':
        env_name = 'Walker2d-v3'
        rtg_target = 5000
        env_d4rl_name = f'walker2d-{config.dataset}-v2'

    elif config.env == 'halfcheetah':
        env_name = 'HalfCheetah-v3'
        rtg_target = 6000
        env_d4rl_name = f'halfcheetah-{config.dataset}-v2'

    elif config.env == 'hopper':
        env_name = 'Hopper-v3'
        rtg_target = 3600
        env_d4rl_name = f'hopper-{config.dataset}-v2'

    else:
        raise NotImplementedError

    dataset_path = f'{config.dataset_dir}/{env_d4rl_name}.pkl'

    wandb.init(project=config.project, group=config.group, name=config.name)

    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(config.device)
    else:
        device = torch.device('cpu')

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    prefix = "dt_" + env_d4rl_name

    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    
    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(config.model_save_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    print(f"device set to: {device}")
    print(f"dataset path: {dataset_path}")
    print(f"model save path: {save_model_path}")
    

    traj_dataset = D4RLDataset(dataset_path, config.context_len, config.rtg_scale)
    traj_dataloader = DataLoader(
        traj_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    data_iter = iter(traj_dataloader)

    state_mean, state_std = traj_dataset.get_state_stats()
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        n_blocks=config.n_blocks,
        hidden_dim=config.embed_dim,
        context_len=config.context_len,
        n_heads=config.n_heads,
        dropout=config.dropout_p,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / config.warmup_steps, 1)
    )

    max_d4rl_score = -1.
    total_updates = 0

    pbar = tqdm(total=config.max_train_iters * config.num_updates_per_iter, position=0)
    pbar.set_description("Training")

    for i_train_iter in range(config.max_train_iters):
        model.train()
        for i_num_update in range(config.num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_dataloader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            
            timesteps = timesteps.to(device)
            states = states.to(device)
            actions = actions.to(device)
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)
            traj_mask = traj_mask.to(device)

            action_target = torch.clone(actions).detach().to(device)

            state_preds, action_preds, return_preds = model.forward(
                					timesteps=timesteps,
                                    states=states,
                                    actions=actions,
                                    returns_to_go=returns_to_go			
            )
            action_preds = action_preds.view(-1, action_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, action_dim)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            total_updates += 1

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            pbar.update(1)
            wandb.log(
                {
                    "action_loss": action_loss.item(),
                    "iter": i_train_iter,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=total_updates
            )

        if (i_train_iter + 1) % config.eval_interval == 0:
            set_seed(config.eval_seed)
            results = evaluate_env(model, device, config.context_len, env, rtg_target, config.rtg_scale,
                                config.n_episodes, config.max_episode_len, state_mean, state_std)

            eval_avg_reward = results['eval/avg_reward']
            eval_avg_length = results['eval/avg_ep_len']
            eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

            wandb.log(
                {
                    "eval/avg_reward": eval_avg_reward,
                    "eval/avg_ep_len": eval_avg_length,
                    "eval/d4rl_score": eval_d4rl_score,
                    "iter": i_train_iter,
                },
                step=total_updates
            )

            if eval_d4rl_score >= max_d4rl_score:
                print("saving max d4rl score model at: " + save_best_model_path)
                torch.save(model.state_dict(), save_best_model_path)
                max_d4rl_score = eval_d4rl_score

        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)


if __name__ == "__main__":
    train()