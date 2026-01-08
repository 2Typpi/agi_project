from typing import Tuple
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange


class RSSMState(torch.Tensor):
    """
    This needs Pytorch >= 1.7.0 otherwise operations on the subclasses do not result in subclassed tensors.
    See https://pytorch.org/docs/main/notes/extending.html#subclassing-torch-tensor for more information. 
    Note that pylance does not like this subclassing and always thinks that all operations yield torch.Tensor objects.
    """

    DET : int
    CAT : int
    LAT : int

    @staticmethod
    def from_data(data, DET, CAT, LAT):

        deter = data[..., :DET]
        stoch = data[..., DET:DET + LAT * CAT]
        logits = data[..., DET + LAT * CAT:].view(*data.shape[:-1], CAT, LAT)
        obj = RSSMState(deter, stoch, logits)
        return obj
    
    @staticmethod
    def __new__(cls, deter, stoch, logits):

        
        # Retrieve the shapes
        batch_size = deter.shape[:-1]
        CAT = logits.shape[-2]
        LAT = logits.shape[-1]
        DET = deter.shape[-1]

        # Flatten the tensors and add a time dimension
        combined_tensor = torch.cat([
            deter.view(*batch_size, DET), 
            stoch.view(*batch_size, LAT * CAT), 
            logits.view(*batch_size, LAT * CAT)
        ], dim=-1)

        # Create an instance of MyTensorDict
        obj = super(RSSMState, cls).__new__(cls, combined_tensor) # type: ignore
        obj.CAT = CAT
        obj.LAT = LAT#logits.shape[-1]
        obj.DET = DET#deter.shape[-1]
        return obj
    

    def __init__(self, deter, stoch, logits):

        
        CAT = logits.shape[-2]
        LAT = logits.shape[-1]
        DET = deter.shape[-1]

        # Initialize attributes to avoid AttributeError

        self.LAT = LAT
        self.CAT = CAT
        self.DET = DET

    def set_deter(self, data):
        
        self[..., :self.DET] = data

    def set_logits(self, data):
        self[..., self.DET + self.LAT * self.CAT:] = data.reshape(*torch.Tensor(self).shape[:-1], -1)
    
    def set_stoch(self, data):
        self[..., self.DET:self.DET + self.LAT * self.CAT] = data

    @property
    def deter(self):
        return self[..., :self.DET].as_subclass(torch.Tensor)  # Extract the deterministic tensor

    @property
    def stoch(self):
        return self[..., self.DET:self.DET + self.LAT * self.CAT].as_subclass(torch.Tensor)  # Extract the stochastic tensor

    @property
    def logits(self):
        return self[..., self.DET + self.LAT * self.CAT:].reshape(*self.shape[:-1], self.CAT, self.LAT).as_subclass(torch.Tensor)  # Extract the logits tensor
    
    @property
    def combined(self):
        return self[..., :self.DET + self.LAT * self.CAT].as_subclass(torch.Tensor) # deterministic + stochastic state
    

    def distribution(self, unimix=0.0):
        """
            Returns a torch distribution object based on the logits stored in the RSSMState object.
        """
        # Important note: The naming is misleading, the logits provided by the prior/posteriors are in fact probabilities!
        if unimix > 0.0:
            probs = (1 - unimix) * self.logits + unimix * torch.ones_like(self.logits) / self.logits.shape[-1]
        else:
            probs = self.logits
        return torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(probs=probs), 1)
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}
        obj = super().__torch_function__(func, types, args, kwargs)
        if isinstance(obj, cls):
            if args:
                if isinstance(args[0], cls): #tensors
                    obj.DET, obj.LAT, obj.CAT = args[0].DET, args[0].LAT, args[0].CAT
                elif isinstance(args[0][0], cls): # lists
                    obj.DET, obj.LAT, obj.CAT = args[0][0].DET, args[0][0].LAT, args[0][0].CAT
        return obj
    
import torch.nn.functional as F
class ConvDecoder(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()

        # First project back to the spatial size used by the encoder's last conv layer
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # -> (3, 64, 64)
            nn.Sigmoid()  # optional: keeps output in [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 4, 4)
        x = self.deconv(x)
        return x
    
class ConvEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()

        # Input: (B, 3, 64, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            # nn.Conv2d(128, 256, 4, stride=2, padding=1),# -> (256, 4, 4)
            # nn.ReLU(),
        )

        self.fc = nn.Linear(128 * 4 * 4, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ImageDecoder(nn.Module):
    def __init__(self, memory_dim, latent, cat):
        super(ImageDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim + latent * cat, 16*16),
            Rearrange("b (c w h) -> b c w h", c = 16, w = 4, h = 4),
            nn.Upsample((16,16)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=(1,1), padding_mode="zeros"),
            nn.ReLU(),
            nn.Upsample((32,32)),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=(1,1), padding_mode="zeros"),
            nn.Sigmoid()
            # nn.Upsample((64,64)),
            # nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=(1,1), padding_mode="zeros"),
            # nn.Sigmoid()
 
        )
    def forward(self, x):
        # Assuming x is of shape (B, 3, W, H)
        x = self.net(x)
        return x
    
    def reconstruction_loss(self, state, observation):
        """
            Computes reconstruction loss between decoded observation (from state) and given observation.
        """
        decoded_img = self.forward(state.combined)
        # RSME loss:
        loss = torch.nn.MSELoss()(decoded_img, observation)
        return loss
    
class ImgDecoder(nn.Module):
    """
        Decodes stochastic and deterministic hidden state into a observation.
    """
    def __init__(self, memory_dim, latent_dim, categoricals) -> None:
        super(ImgDecoder, self).__init__()
        self.decoder = ConvDecoder(input_dim=latent_dim*categoricals + memory_dim)

    def forward(self, state_combined):
        img = self.decoder(state_combined)
        return img
    
    def reconstruction_loss(self, state, observation):
        """
            Computes reconstruction loss between decoded observation (from state) and given observation.
        """
        decoded_img = self.forward(state.combined)
        loss = torch.nn.MSELoss()(decoded_img, observation)
        return loss

class ImgEncoder(nn.Module):
    """
        Encodes observations into a stochastic representation (referred to as posterior) based on a deterministic hidden state.
    """
    base_config = {
            "hidden_dim" : 256,
            "n_layers" : 1,
            "activation" : nn.ReLU()
        }
    
    def __init__(self, latent_img_dim, state_shape,  latent_dim, categoricals, img_enc=None,config=base_config) -> None:
        super(ImgEncoder, self).__init__()
        
        activation = config["activation"]
        hidden_dim = config["hidden_dim"]

        # encode images using cnn:
        self.img_enc = ConvEncoder(output_dim=latent_img_dim) if img_enc is None else img_enc
        self.encoder = nn.Sequential(

            nn.Linear(latent_img_dim + state_shape, hidden_dim),
            activation,
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                            activation) for i in range(config["n_layers"])],
            nn.Linear(hidden_dim, latent_dim*categoricals)
        )


        self.latent_dim = latent_dim
        self.categoricals = categoricals
        
    def forward(self, observation, state : RSSMState):
        """
            Computes the logits.
        """
        encoded_img = self.img_enc(observation)
        input = torch.cat((encoded_img, state.deter), dim=1)
        enc = self.encoder(input)
        enc = enc.reshape((-1, self.categoricals, self.latent_dim))
        return enc
    
    
    def predict(self, observation, state : RSSMState):
        """
            Predicts stochastic representation, stored alongside the used deterministic state in a new RSSMState object.
        """
        enc = self.forward(observation, state)
        probs = torch.softmax(enc, dim=-1)

        distribution = torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(probs=probs), 1)
        discretized = distribution.rsample()
        new_state = RSSMState(stoch=torch.flatten(discretized,start_dim=1), logits=probs, deter=state.deter)
        return new_state
    
class Encoder(nn.Module):
    """
        Encodes observations into a stochastic representation (referred to as posterior) based on a deterministic hidden state.
    """
    base_config = {
            "hidden_dim" : 256,
            "n_layers" : 1,
            "activation" : nn.ReLU()
        }
    
    def __init__(self, observation_shape, state_shape,  latent_dim, categoricals, config=base_config) -> None:
        super(Encoder, self).__init__()
        
        activation = config["activation"]
        hidden_dim = config["hidden_dim"]


        self.encoder = nn.Sequential(
            nn.Linear(observation_shape + state_shape, hidden_dim),
            activation,
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                            activation) for i in range(config["n_layers"])],
            nn.Linear(hidden_dim, latent_dim*categoricals)
        )


        self.latent_dim = latent_dim
        self.categoricals = categoricals
        
    def forward(self, observation, state : RSSMState):
        """
            Computes the logits.
        """
        input = torch.cat((observation, state.deter), dim=1)
        enc = self.encoder(input)
        enc = enc.reshape((-1, self.categoricals, self.latent_dim))
        return enc
    
    
    def predict(self, observation, state : RSSMState):
        """
            Predicts stochastic representation, stored alongside the used deterministic state in a new RSSMState object.
        """
        enc = self.forward(observation, state)
        probs = torch.softmax(enc, dim=-1)

        distribution = torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(probs=probs), 1)
        discretized = distribution.rsample()
        new_state = RSSMState(stoch=torch.flatten(discretized,start_dim=1), logits=probs, deter=state.deter)
        return new_state


class DynamicsPredictor(nn.Module):
    """
        Predicts a stochastic representation (referred to as prior) based on the deterministic hidden state.
    """
    base_config = {
        "hidden_dim" : 256,
        "n_layers" : 1,
        "activation" : nn.ReLU()
    }
    def __init__(self, input_shape, latent_dim, categoricals, config=base_config) -> None:
        super(DynamicsPredictor, self).__init__()

        activation = config["activation"]
        hidden_dim = config["hidden_dim"]


        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            activation,
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                            activation) for i in range(config["n_layers"])],
            nn.Linear(hidden_dim, latent_dim*categoricals)
        )

        self.latent_dim = latent_dim
        self.categoricals = categoricals

    def forward(self, state : RSSMState):
        """
            Computes the logits.
        """
        enc = self.encoder(state.deter)
        enc = enc.reshape((-1, self.categoricals, self.latent_dim))
        return enc

    def predict(self, state : RSSMState) -> RSSMState:
        """
            Predicts stochastic representation (stored in RSSMState object alongside used determnisitic hidden state) based on determninistic hidden state stored in RSSMState object.
        """
        
        enc = self.forward(state)
        probs = torch.softmax(enc, dim=-1)
        distribution = torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(probs=probs), 1)


        discretized = distribution.rsample()#torch.nn.functional.one_hot(distribution.sample(), num_classes=self.latent_dim).float()
        new_state = RSSMState(stoch=torch.flatten(discretized,start_dim=1), logits=probs, deter=state.deter)
        return new_state
    

class Decoder(nn.Module):
    """
        Decodes stochastic and deterministic hidden state into a observation.
    """
    def __init__(self, observation_shape, memory_dim, latent_dim, categoricals, config) -> None:
        hidden_dim = 128
        use_bias = True
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim*categoricals + memory_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_shape, bias=use_bias)
        )
        self.output_shape = observation_shape

    def forward(self, state_combined):
        img = self.decoder(state_combined)
        return img
    
class SequenceModel(nn.Module):
    """
        Predicts new determninistic hidden state based on stochastic hidden state, action and old deterministic hidden state.
    """
    base_config = {
            "hidden_dim" : 256,
            "n_layers" : 1,
            "activation" : nn.ReLU(),
        },
    
    def __init__(self, latent_size, action_size, memory_size, config=base_config) -> None:
        super(SequenceModel, self).__init__()


        self.memory_size = memory_size
        activation = config["activation"]
        hidden_dim = config["hidden_dim"]


        self.linear = nn.Sequential(
            nn.Linear(latent_size+action_size, hidden_dim),
            activation,
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                            activation) for i in range(config["n_layers"])]
        )

        self.memory = nn.GRU(input_size=hidden_dim, hidden_size=memory_size, num_layers=1, batch_first=True)


    def forward_seq(self, actions, states : RSSMState) -> RSSMState:
        #actions = torch.clamp(actions, -1.0, 1.0)
        B,T,_ = actions.shape

        inpt = torch.cat((states.stoch[:, :-1], actions), dim=-1) # last stoch is not important, as before it was computed from the last state i.e. after doing the last aciton
        inpt = inpt.reshape((B*T, inpt.shape[-1]))

        x = self.linear(inpt).reshape((B,T, -1))
        hidden_state = states.deter[:,0].reshape((1, B, states.deter.shape[-1]))[:, :, :self.memory_size].contiguous()
        new_hidden_state = self.memory(x, hidden_state)[0]
        return RSSMState(stoch=states.stoch[:, 1:], logits=states.logits[:, 1:], deter=new_hidden_state) #skip the first parts of the state (stochastic and logits), as they belong to the previous state and do not correspond to current actions & observation

    def forward(self, action, state : RSSMState) -> RSSMState:
        #action = torch.clamp(action, -1.0, 1.0)

        x = self.linear(torch.cat((state.stoch, action), dim=-1)).unsqueeze(1)
        hidden_state = state.deter.unsqueeze(0)[:,:,:self.memory_size].contiguous()
        new_hidden_state = self.memory(x, hidden_state)[1].squeeze(0)
        return RSSMState(stoch=state.stoch, logits=state.logits, deter=new_hidden_state)
    




class RSSM(nn.Module):
    """
        Recurrent state-space model as used in Dreamer. Note that the Decoder is also part of this model.

    """
    base_config = {
        "sequence_model" : {
            "hidden_dim" : 256,
            "n_layers" : 1,
            "activation" : nn.ReLU(),

        },
        "dynamics_predictor" : {
            "hidden_dim" : 256,
            "n_layers" : 1,
            "activation" : nn.ReLU(),
        },


    }

    def __init__(self, encoder, decoder, device, latent_space, action_space, sequence_model_memory_size, config=base_config) -> None:
    
        super(RSSM, self).__init__()

            
        self.latent_size = latent_space[0] * latent_space[1]
        self.latent_space = latent_space
        self.memory_dim = sequence_model_memory_size
 
        self.encoder = encoder
        self.decoder = decoder


        # Sequence model: h = f(h, z, a)
        self.sequence_model = SequenceModel(latent_size= self.latent_size, action_size = action_space, memory_size=self.memory_dim, config=config["sequence_model"])#, num_models=4)#, num_models=5)
        
        # Dynamics predictor: z ~ p(z|h)
        self.dynamics_predictor = DynamicsPredictor(input_shape=self.memory_dim, latent_dim=self.latent_space[0], categoricals=self.latent_space[1], config=config["dynamics_predictor"])
        
        
        self.device = device

    def reconstruction_loss(self, state : RSSMState, observation):
        """
            Computes reconstruction loss between decoded observation (from state) and given observation.
        """
        rec_loss = self.decoder.reconstruction_loss(state.reshape((-1, state.shape[-1])), observation.reshape((-1, *observation.shape[2:])))
        return rec_loss, {"rec_loss" : rec_loss.detach()}
    
    def get_initial_state(self, batch_size) -> RSSMState:
        """
            Returns an initial state (stochastic + deterministic), initialized as zeros.
        """
        device = self.device
        state = RSSMState(stoch=torch.zeros((batch_size, self.latent_size), device=device), logits=torch.zeros((batch_size, self.latent_space[0], self.latent_space[1]), device=device), deter=torch.zeros((batch_size, self.memory_dim),device=device))
        return state


    def imagine(self, prev_action, prev_state : RSSMState) -> RSSMState:
        """
            Performs one 'dream' step i.e. step in the sequence model followed by a prediction of the new state.
        """
        deter_state = self.sequence_model(prev_action, prev_state) # also contains stochastic state from prev_state
        prior_state = self.dynamics_predictor.predict(deter_state)
        return prior_state
    


    def observe(self,  observation, prev_action, prev_state : RSSMState) -> Tuple[RSSMState, RSSMState]:
        """
            Performs one step in the model while observing the observation. Returns both, the computed prior state (stochastic state based on the dynamics model) as well as 
            posterior state (stochastic state obtained by using the observation).
        """
        # action: a^{t-1}, prev_state: (h^{t-1}, z^{t-1}),
        
        # Do step in sequence model

        deter_state = self.sequence_model(prev_action, prev_state)
        # Compute prior state based on hidden state
        prior_state = self.dynamics_predictor.predict(deter_state) # uses only h^t => \hat{z}^t 

        # Computer posterior state based on hidden state and observation
        posterior_state = self.encoder.predict(observation, deter_state)

        return prior_state, posterior_state
    
   
    def observe_posterior(self,  observation, prev_action, prev_state : RSSMState) -> RSSMState:
        """
            Performs one step in the model while observing the observation. Returns the 
            posterior state (stochastic state obtained by using the observation).
        """
        # action: a^{t-1}, prev_state: (h^{t-1}, z^{t-1}),
        
        # Do step in sequence model

        deter_state = self.sequence_model(prev_action, prev_state)
   
        # Computer posterior state based on hidden state and observation
        posterior_state = self.encoder.predict(observation, deter_state)

        return posterior_state
        
    def rollout_imagination_actions(self, actions, prev_state : RSSMState):
        # actions: (B,T,AS)
        states = []
        state = prev_state
        horizon = actions.shape[1]
        for t in range(horizon):
            states.append(state)
            state = self.imagine(actions[:,t,:], state)
        return torch.stack(states, dim=1)
    
    def rollout_imagination(self, horizon:int, policy:nn.Module, prev_state : RSSMState) -> Tuple[RSSMState, torch.Tensor]:
        """
            Performs T steps of dreaming, starting from a initial state and using a given policy.
        """
        states = []
        actions = []
        state = prev_state
        for t in range(horizon):
            action = policy.get_action(state) # <--- here was a bug which took days to solve
            states.append(state)
            actions.append(action)
            state = self.imagine(action, state)
   
        return torch.stack(states, dim=1), torch.stack(actions, dim=1)  # type: ignore

    def rollout_observation(self, seq_len:int, observations, actions, prev_state : RSSMState)-> Tuple[RSSMState, RSSMState]:


        """
            Peforms T steps in the world model while observing the observation starting from an initial state. Actions are given as a list. 
            Returns a sequence over prior and posterior states.
        """
        priors = []
        posteriors = []
        for t in range(0, seq_len):
            prior_state, posterior_state = self.observe(observations[:,t,:], actions[:,t,:], prev_state)
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        return torch.stack(priors,dim=1), torch.stack(posteriors,dim=1) # type: ignore
    
    
    def rollout_observation_posterior(self, seq_len:int, observations, prev_actions, prev_state : RSSMState)-> RSSMState:


        """
            Peforms T steps in the world model while observing the observation starting from an initial state. Actions are given as a list. 
            Returns a sequence over prior and posterior states.
        """
        posteriors = []
        for t in range(0, seq_len):
            posterior_state = self.observe_posterior(observations[:,t,:], prev_actions[:,t,:], prev_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        return torch.stack(posteriors,dim=1) # type: ignore
    


### Energy based RSSM


class JointObservationStateEmbedding(nn.Module):
    base_config = {
            "hidden_dim" : 256,
            "n_layers" : 0,
            "activation" : nn.ReLU(),
            "normalization" : None,
            "energy_loss" : "bce"
        }
    def __init__(self, o_shape, h_shape, config=base_config):
        super(JointObservationStateEmbedding, self).__init__()

        activation = config["activation"]
        hidden_dim = config["hidden_dim"]
        energy_loss = config["energy_loss"]
        if config["normalization"] == "layernorm":
            norm = lambda x : nn.LayerNorm(x)
        else:
            norm = lambda x : nn.Identity()

        self.energy = nn.Sequential(
            nn.Linear(o_shape + h_shape, hidden_dim),
            norm(hidden_dim),
            activation,
            *[nn.Sequential(nn.Linear(hidden_dim // (2**i), hidden_dim // (2**(i+1))),
                            norm(hidden_dim // (2**(i+1))),
                            activation) for i in range(config["n_layers"])],
            nn.Linear(hidden_dim // (2**(config["n_layers"])), 1)
        )
        self.energy_loss = energy_loss

    def forward(self, state : RSSMState, observation):

        input = torch.cat([observation, state.stoch, state.deter], dim=1)
        energy = self.energy(input)
        return energy
    
    def contrastive_loss(self, states : RSSMState, observations, filter=None):
        #states: B, T, F
        B, T, F = states.deter.shape
        shuffled_idx = torch.randperm(B)
        shuffled_states = states[shuffled_idx]
        
        positive = self.forward(states.reshape((-1, states.shape[-1])), observations.reshape((-1, *observations.shape[2:]))) # type: ignore
        negative = self.forward(shuffled_states.reshape((-1, states.shape[-1])), observations.reshape((-1, *observations.shape[2:]))) # type: ignore
        d = {}
        if filter is not None:
            selection = torch.sigmoid(positive) < filter
            positive = positive[selection]
            negative = negative[selection]
            d["cd_ratio"] = selection.sum(dim=0) / (B*T)

        if self.energy_loss == "bce":
            positive_loss = torch.nn.BCEWithLogitsLoss()(positive, torch.ones_like(positive)) # state matches observation => low energy
            negative_loss = torch.nn.BCEWithLogitsLoss()(negative, torch.zeros_like(negative)) # state does not match observation => high energy
        elif self.energy_loss == "mse_reg":
            #positive = torch.clamp(positive, -100.0, 100.0)
            #negative = torch.clamp(negative, -100.0, 100.0)
            positive_loss = torch.nn.BCEWithLogitsLoss()(positive, torch.ones_like(positive)) # state matches observation => low energy
            negative_loss = torch.nn.BCEWithLogitsLoss()(negative, 0.01*torch.ones_like(negative))
        elif self.energy_loss == "mse":
            positive_loss = torch.mean((positive - 1.0)**2) # state matches observation => low energy
            negative_loss = torch.mean((negative + 1.0)**2) # state does not match observation => high energy
        else:
            raise NotImplementedError
        return_dict = {"positive_loss" : positive_loss.detach(), "negative_loss" : negative_loss.detach(), "positive_energy" : positive.mean().detach(), "negative_energy" : negative.mean().detach()}
            
        return positive_loss + negative_loss, return_dict | d#positive_loss, negative_loss, positive
    
class ImageJointObservationStateEmbedding(JointObservationStateEmbedding):
    base_config = {
            "hidden_dim" : 256,
            "n_layers" : 0,
            "activation" : nn.ReLU(),
            "normalization" : None,
            "energy_loss" : "bce"
        }
    def __init__(self, o_shape, h_shape, observation_encoder, config=base_config):
        super().__init__(o_shape, h_shape, config)
        self.observation_encoder = observation_encoder

    def forward(self, state : RSSMState, observation):
        observation = self.observation_encoder(observation)
        return super().forward(state, observation)

class EnergyRSSM(RSSM):
    """
        Recurrent state-space model as used in Dreamer. Note that the Decoder is also part of this model.

    """
    def __init__(self, encoder, decoder, energy_model, device, latent_space, action_space, sequence_model_memory_size) -> None:
        super(EnergyRSSM, self).__init__(encoder, decoder, device, latent_space, action_space, sequence_model_memory_size)
        self.energy_model = energy_model
        
    def reconstruction_loss(self, state : RSSMState, observation):
        """
            Computes reconstruction loss between decoded observation (from state) and given observation.
        """
        rec_loss, rec_dict = super().reconstruction_loss(state.detach(), observation)
        energy_loss, energy_dict = self.energy_model.contrastive_loss(state, observation)
        
        return rec_loss + energy_loss, rec_dict | energy_dict