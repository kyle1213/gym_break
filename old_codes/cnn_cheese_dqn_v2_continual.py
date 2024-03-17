import gymnasium as gym
import numpy as np
import random
import datetime
import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torchvision.transforms as transforms

# from baselines.common.atari_wrappers import make_atari, wrap_deepmind


# DQN을 위한 파라미터 값 세팅
state_size = [95, 144, 1]
action_size = 4

load_model = True
train_mode = True

batch_size = 128
mem_maxlen = 100000  # replay memory length
discount_factor = 0.99
learning_rate = 0.0001 * 0.1

run_step = 10000000 if train_mode else 0
test_step = 0 if train_mode else 10000
train_start_step = 0

print_interval = 10
save_interval = 100

epsilon_eval = 0
epsilon_init = 0.01 if train_mode else epsilon_eval
epsilon_min = 0.01
explore_step = 1  # 천만중의 3.5퍼만 eps move = 35만
epsilon_delta = (epsilon_init - epsilon_min) / (1 * explore_step) if train_mode else 0.

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/cnn_cheese_dqn10_v2_continual/{date_time}"
load_path = f"./saved_models/cnn_cheese_dqn10_v2/20231207135600 - a"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transforms:
    def to_gray(frame1, frame2=None):
        # state frame gray로 0~1로 값 고정 및 프레임 자르기
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175, 150)),
            transforms.ToTensor()
        ])
        # 블럭이 없고 ball과 bar가 있는 곳만 사용하며, 이전 프레임과 차를 통해 공의 이동 방향 나타내기, 현재 공 or bar = 1, 이전 공 or bar = -1, else = 0
        if frame2 is not None:
            new_frame = torch.sign(
                torch.sign(gray_transform(frame1)[:, 80:, 3:-3]) - 0.5 * torch.where(frame2 > 0, frame2,
                                                                                     torch.tensor(0.)))
        else:
            new_frame = torch.sign(gray_transform(frame1)[:, 80:, 3:-3]).float()
        return new_frame


# 초기 env make 및 첫 state transform
def init_gym_env(env_path):
    env = gym.make(env_path, render_mode='human')
    env.metadata['render_fps'] = 60
    state = env.reset()[0]
    state = Transforms.to_gray(state)

    return env, state


# cnn, FC layer 기반의 DQN 모델 사용
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32,
                                     kernel_size=(8, 8), stride=(4, 4))
        self.conv_bn1 = torch.nn.BatchNorm2d(32)
        dim1 = ((state_size[0] - 8) // 4 + 1, (state_size[1] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=(4, 4), stride=(2, 2))
        self.conv_bn2 = torch.nn.BatchNorm2d(64)
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=(3, 3), stride=(1, 1))
        self.conv_bn3 = torch.nn.BatchNorm2d(64)
        dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)
        self.fc1 = torch.nn.Linear(64 * dim3[0] * dim3[1], 128)
        self.fc2 = torch.nn.Linear(128, 4)
        self.bn1 = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.conv_bn1(self.conv1(x)))
        x = F.relu(self.conv_bn2(self.conv2(x)))
        x = F.relu(self.conv_bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


class DQNAgent():
    def __init__(self):
        self.network = DQN().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path)
        self.epsilon = epsilon_init

        if load_model:
            print(f"... Load Model from {load_path}/model.ckpt ...")
            checkpoint = torch.load(load_path + "/model.ckpt", map_location=device)
            self.network.load_state_dict(checkpoint["dqn"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.network.train(False)
        epsilon = self.epsilon if training else epsilon_eval
        # 랜덤하게 행동 결정
        if epsilon > random.random():
            idx = np.random.randint(0, action_size, size=1)[0]
            action = [0] * action_size
            action[idx] = 1
            action = torch.Tensor(action)
        # 네트워크 연산에 따라 행동 결정
        else:
            action = self.network(state.unsqueeze(dim=0).to(device))
        self.network.train(training)
        return action.squeeze(0)

    def append_sample(self, state, action, reward, next_state, done, action_idx):
        self.memory.append((state, action, reward, next_state, done, action_idx))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state = np.stack([b[0] for b in batch], axis=0)
        action = np.stack([b[1] for b in batch], axis=0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0)
        action_idx = torch.stack([b[5] for b in batch])

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                      [state, action, reward, next_state, done])

        next_q = self.network(next_state)
        target_q = reward.unsqueeze(1) + ((1 - done) * discount_factor * next_q.max(1, keepdims=True).values)
        q = self.network(state)
        q = q.gather(1, action_idx.unsqueeze(1))

        loss = F.huber_loss(target_q.detach(), q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 엡실론 감소
        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)

        return loss.item()

    def save_model(self):
        print(f" ... Save Model to {save_path}/ckpt ...")
        torch.save({"dqn": self.network.state_dict(), "optimizer": self.optimizer.state_dict()},
                   save_path + '/model.ckpt')

    def write_summary(self, score, loss, step, fake_score):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/fake_score", fake_score, step)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == '__main__':
    # env init
    env, state = init_gym_env('ALE/Breakout-v5')

    # agent init
    agent = DQNAgent()

    # values init
    scores, episode, losses, fake_scores = [], 0, [], []
    score, fake_score = 0, 0
    lives = 5

    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False

        action = agent.get_action(state, train_mode)
        action_idx = torch.argmax(action)

        old_state = state
        state, real_reward, done, _, info = env.step(action_idx)
        state = Transforms.to_gray(state, old_state)
        env.render()

        fake_reward = real_reward * 10

        if torch.max(torch.sum(state, dim=1)) >= 6:  # bar와 ball이 겹치면==8, bar==4, ball==4, old object=-4
            fake_reward += 1
        if torch.min(torch.sum(state, dim=1)) <= -6:
            fake_reward += 1
        fake_score += fake_reward

        score += real_reward

        if train_mode:
            agent.append_sample(old_state, action.data.cpu().numpy(), fake_reward, state, [done], action_idx.to(device))

        if train_mode and step > max(batch_size, train_start_step):
            # 학습 수행
            loss = agent.train_model()
            losses.append(loss)

        if done:
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
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score, mean_loss, step, mean_fake_score)
                scores, losses, fake_scores = [], [], []

                print(
                    f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / fake Score: {mean_fake_score:.2f} /" + \
                    f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()