import gymnasium as gym
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.distributions import Categorical


# DQN을 위한 파라미터 값 세팅
state_size = [95, 144, 1]
action_size = 4

load_model = False
train_mode = True

discount_factor = 0.99
learning_rate = 0.0001

run_step = 10000000 if train_mode else 0
test_step = 0 if train_mode else 10000
target_update_step = 5000

print_interval = 10
save_interval = 100


# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/AC_batch_cheese_cnn/{date_time}"
load_path = f"./saved_models/AC_batch/20231115155738"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175, 150)),
            transforms.ToTensor()
        ])
        # Subtract one frame from the other to get sense of ball and paddle direction
        if frame2 is not None:
            new_frame = torch.sign(
                torch.sign(gray_transform(frame1)[:, 80:, 3:-3]) - 0.5 * torch.where(frame2 > 0, frame2,
                                                                                     torch.tensor(0.)))
        else:
            new_frame = torch.sign(gray_transform(frame1)[:, 80:, 3:-3]).float()
        return new_frame


def init_gym_env(env_path):
    env = gym.make(env_path, render_mode='human')
    env.metadata['render_fps'] = 60
    state = env.reset()[0]
    state = Transforms.to_gray(state)

    return env, state


class Actor_Critic(torch.nn.Module):
    def __init__(self):
        super(Actor_Critic, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32,
                                         kernel_size=(8, 8), stride=(4, 4))
        dim1 = ((state_size[0] - 8) // 4 + 1, (state_size[1] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                         kernel_size=(4, 4), stride=(2, 2))
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                         kernel_size=(3, 3), stride=(1, 1))
        dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

        self.fc1 = torch.nn.Linear(64 * dim3[0] * dim3[1], 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.actor = torch.nn.Linear(128, 4)
        self.critic = torch.nn.Linear(128, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.softmax(self.actor(x), 1), self.critic(x)


class ACAgent():
    def __init__(self):
        self.actor_critic = Actor_Critic().to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate, weight_decay=0.9)
        self.data = []
        self.writer = SummaryWriter(save_path) if load_model == False else None

        if load_model:
            print(f"... Load Model from {load_path}/model.ckpt ...")
            checkpoint = torch.load(load_path + "/model.ckpt", map_location=device)
            self.actor_critic.load_state_dict(checkpoint["actor_critic"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.actor_critic.train(False)
        action, _ = self.actor_critic(state.unsqueeze(dim=0).to(device))
        self.actor_critic.train(training)
        return action.squeeze(0)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        state = np.stack([b[0] for b in self.data], axis=0)
        action = np.stack([b[1].detach().cpu() for b in self.data], axis=0)
        reward = np.stack([b[2] for b in self.data], axis=0)
        next_state = np.stack([b[3] for b in self.data], axis=0)
        done = np.stack([b[4] for b in self.data], axis=0)
        action_idx = torch.stack([b[5] for b in self.data])

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                      [state, action, reward, next_state, done])
        self.data = []
        return state, action, reward, next_state, done, action_idx

    def train_model(self):
        state, action, reward, next_state, done, action_idx = self.make_batch()

        _, next_v = self.actor_critic(next_state)
        target_v = reward[0] + ((1 - done[0]) * discount_factor * next_v)
        action_pred, v = self.actor_critic(state)
        critic_loss = F.huber_loss(target_v.detach(), v)

        advantage = target_v - v
        selected_prob = action_pred.gather(-1, action_idx.unsqueeze(0))
        selected_log_prob = torch.log(selected_prob)
        actor_loss = (-selected_log_prob * advantage.detach()).mean()
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save_model(self):
        print(f" ... Save Model to {save_path}/ckpt ...")
        torch.save({"actor_critic": self.actor_critic.state_dict(), "optimizer": self.optimizer.state_dict()},
                   save_path + '/model.ckpt')

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)


os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == '__main__':
    # env init
    env, state = init_gym_env('ALE/Breakout-v5')

    # agent init
    agent = ACAgent()

    # values init
    scores, episode, losses, fake_scores, actor_losses, critic_losses = [], 0, [], [], [], []
    score, fake_score = 0, 0
    lives = 5

    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False

        action = agent.get_action(state, train_mode)
        m = Categorical(action)
        action_idx = m.sample()

        old_state = state
        state, real_reward, done, _, info = env.step(action_idx)
        state = Transforms.to_gray(state, old_state)
        env.render()

        fake_reward = real_reward

        if torch.max(torch.sum(state, dim=1)) >= 6:  # bar와 ball이 겹치면==8, bar==4, ball==4, old object=-4
            fake_reward += 1
        if torch.min(torch.sum(state, dim=1)) <= -6:
            fake_reward += 1
        fake_score += fake_reward

        score += real_reward

        if train_mode:
            agent.put_data((old_state, action, fake_reward, state, done, action_idx))

        if done:
            actor_loss, critic_loss = agent.train_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            losses.append(actor_loss + critic_loss)

            episode += 1
            state = env.reset()[0]
            state = Transforms.to_gray(state)
            lives = 5
            scores.append(score)
            fake_scores.append(fake_score)
            score, fake_score = 0, 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_fake_score = np.mean(fake_scores)
                mean_actor_loss = np.mean(actor_losses)
                mean_critic_loss = np.mean(critic_losses)
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores, losses, fake_scores = [], [], [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / fake Score: {mean_fake_score:.2f} / " + \
                      f"actor Loss: {mean_actor_loss:.4f} / critic Loss: {mean_critic_loss:.4f} / Loss: {mean_loss:.4f}")

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()
