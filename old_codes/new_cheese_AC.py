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

batch_size = 128
mem_maxlen = 100000
discount_factor = 0.99
learning_rate = 0.0001

run_step = 10000000 if train_mode else 0
test_step = 0 if train_mode else 10000
train_start_step = 0

print_interval = 10
save_interval = 100

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/mlp_cheese_ac/{date_time}"
load_path = f"./saved_models/AC/20231115155738"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175,150)),
            transforms.ToTensor()
        ])
        # Subtract one frame from the other to get sense of ball and paddle direction
        if frame2 is not None:
            new_frame = torch.sign(torch.sign(gray_transform(frame1)[:, 80:, 3:-3]) - 0.5*torch.where(frame2 > 0, frame2, torch.tensor(0.)))
        else:
            new_frame = torch.sign(gray_transform(frame1)[:, 80:, 3:-3]).float()
        state = torch.sum(new_frame, dim=1)
        return new_frame, state.squeeze(0)


def init_gym_env(env_path):
    env = gym.make(env_path, render_mode='human')
    env.metadata['render_fps'] = 60
    state = env.reset()[0]
    frame, state = Transforms.to_gray(state)

    return env, frame, state


class Actor_Critic(torch.nn.Module):
    def __init__(self):
        super(Actor_Critic, self).__init__()
        self.fc1 = torch.nn.Linear(144, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.actor = torch.nn.Linear(128, 4)
        self.critic = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.softmax(self.actor(x), 1), self.critic(x)


class ACAgent():
    def __init__(self):
        self.actor_critic = Actor_Critic().to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate, weight_decay=0.9)
        self.writer = SummaryWriter(save_path) if load_model == False else None
        self.data = []

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

    def write_summary(self, score, actor_loss, critic_loss, step, fake_score):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)
        self.writer.add_scalar("model/fake_score", fake_score, step)


os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == '__main__':
    # env init
    env, frame, state = init_gym_env('ALE/Breakout-v5')

    # agent init
    agent = ACAgent()

    # values init
    actor_losses, critic_losses, scores, episode, losses, fake_scores = [], [], [], 0, [], []
    score, fake_score = 0, 0
    lives = 5

    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False

        #plt.imshow(torch.permute(frame, (1,2,0)))
        #plt.show()
        #plt.imshow(state)
        #plt.show()

        action = agent.get_action(state, train_mode)
        m = Categorical(action)
        action_idx = m.sample()

        old_state = state
        old_frame = frame
        frame, real_reward, done, _, info = env.step(action_idx)
        frame, state = Transforms.to_gray(frame, old_frame)
        env.render()

        fake_reward = real_reward

        if torch.max(state) >= 6:  # bar와 ball이 겹치면==8, bar==4, ball==4, old object=-4
            fake_reward += 1
        if torch.min(state) <= -6:
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
            frame = env.reset()[0]
            frame, state = Transforms.to_gray(frame)
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
                agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step, mean_fake_score)
                actor_losses, critic_losses, scores, losses, fake_scores = [], [], [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / fake Score: {mean_fake_score:.2f} /" + \
                      f"actor Loss: {mean_actor_loss:.4f} / critic Loss: {mean_critic_loss:.4f} / Loss: {mean_loss:.4f}")

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()