import torch

from rl.dreamer.rssm import RSSMState


class DreamerAgent:
    def __init__(self, dreamer, policy, device):
        self.dreamer = dreamer
        self.policy = policy
        self.device = device
    
    def get_action(self, observation, goal=None):
        return self.policy.get_action(self.dreamer.state.combined, goal)
    def step(self, action, next_observation):
        return self.dreamer.step(action, next_observation)
    def update(self, buffer, step):
        return self.dreamer.update(buffer, step)
    

class DreamerShapeEnvPrediction(torch.nn.Module):
    def __init__(self, class_sizes, names, DET,CAT,LAT):
        super(DreamerShapeEnvPrediction, self).__init__()
        self.DET = DET
        self.CAT = CAT
        self.LAT = LAT
        self.net = torch.nn.Sequential(
            torch.nn.Linear(DET, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, sum(class_sizes)),  # predict size, color, shape
        )
        self.names = names
        self.class_sizes = class_sizes
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
    def predict(self, state):
        logits = self.net(state)
        logits = torch.split_with_sizes(logits, self.class_sizes, dim=-1)
        dists = []
        for i in range(len(logits)):
            dist_i = torch.distributions.OneHotCategorical(logits=logits[i])
            dists.append(dist_i)
        return dists
    def update(self, buffer, step):
        with torch.no_grad():
            data = buffer.sample(256,1)
            if data is None:
                return {}
            state = RSSMState.from_data(data["state"][:,0], self.DET,self.CAT,self.LAT)

        dists = self.predict(state.deter.detach())
        loss = 0
        loss_dict = {}
        for i in range(len(self.class_sizes)):
            # label to one-hot:
            #print(data[f"label_{i}"][:,0].long().shape, dists[i].probs.shape)
            loss += -dists[i].log_prob(data[f"label_{self.names[i]}"][:,0].long()).mean()
            loss_dict[f"class_{self.names[i]}_loss"] = loss.detach()

            # accuracy:
            acc = (dists[i].probs.argmax(dim=-1) == data[f"label_{self.names[i]}"][:,0].long().argmax(dim=-1)).float().mean()
            loss_dict[f"class_{self.names[i]}_acc"] = acc.detach()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss_dict