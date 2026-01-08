import torch
import torch.nn as nn
from torch.distributions.utils import probs_to_logits



from rl.dreamer.rssm import RSSM, RSSMState

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)+1e-6)
        return loss

class Dreamer(nn.Module):
    """
        Creates a dreamer based virtual environment over a base environment using a learned world model. The base environment can be again a virtual environment or a real environment.

    """

    
    def __init__(self, rssm, latent_space, sequence_model_memory_size, device, batch_size, unroll_steps, free_bits, update_initial_state="observed", reset_initial_state="zero", unimix=0.01) -> None:
        super(Dreamer, self).__init__()
        

        self.rssm = rssm.to(device)
        # Running observation and hidden state
        self.state = self.rssm.get_initial_state(batch_size=1)
        self.device = device
        self.DYN_SCALE = 0.5
        self.REP_SCALE = 0.1
        self.DET = sequence_model_memory_size
        self.CAT=latent_space[0]
        self.LAT=latent_space[1]
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.free_bits = free_bits
        self.update_initial_state = update_initial_state
        self.reset_initial_state = reset_initial_state
        self.opt = torch.optim.Adam(params=self.rssm.parameters(), lr=1e-3)
        self.unimix = unimix
        
    def reset(self, buffer=None, observation=None):
        if self.reset_initial_state == "zero":
            self.state = self.rssm.get_initial_state(batch_size=1)
        else:
            raise NotImplementedError
        
    def step(self, action, next_observation) -> RSSMState:
        with torch.no_grad():
            posterior_state = self.rssm.observe_posterior(next_observation, action, self.state)
            self.state = posterior_state
            return posterior_state
        
    def observe_step(self, action, next_observation, state) -> RSSMState:
        with torch.no_grad():
            posterior_state = self.rssm.observe_posterior(next_observation, action, state)
            return posterior_state
        
    def observe(self, observation) -> RSSMState:
        """
            Encode the current observation using the current state. Obtain a new state.
        """
        self.state = self.rssm.encoder.predict(observation, self.state)
        return self.state
    
    def encode_observation(self, observation, state : RSSMState) -> RSSMState:
        """
            Encodes an observation based on the current deterministic state.
        """
        posterior_state = self.rssm.encoder.predict(observation, state) # type: ignore
        return posterior_state
    
    def dream_step(self, action, prev_state : RSSMState) -> RSSMState:
        """
            Performs a dreamed step in the world model.
        """
        return self.rssm.imagine(action, prev_state)
    
    def dream_actions(self, prev_actions, prev_state : RSSMState) -> RSSMState:
        return self.rssm.rollout_imagination_actions(prev_actions, prev_state)
        
    def dream(self, horizon:int, policy:nn.Module, prev_state : RSSMState):
        """
            Performs T dreamed steps in the world model.
        """
        return self.rssm.rollout_imagination(horizon, policy, prev_state)
    
    
    def observe_sequence_prior(self, prev_actions, observations, prev_state : RSSMState) -> RSSMState:
        """
            Encode observation, action sequence into sequence of posterior model states
        """
        prior_states, posterior_states = self.rssm.rollout_observation(seq_len = prev_actions.shape[1], observations=observations, prev_actions=prev_actions, prev_state = prev_state)
        return prior_states
    
    def observe_sequence(self, prev_actions, observations, prev_state : RSSMState) -> RSSMState:
        """
            Encode observation, action sequence into sequence of posterior model states
        """
        posterior_states = self.rssm.rollout_observation_posterior(seq_len = prev_actions.shape[1], observations=observations, prev_actions=prev_actions, prev_state = prev_state)
        return posterior_states
    
    def uniform_mix(self, logits, unimix: float = 0.01):
        if unimix > 0.0:
            # compute probs from the logits
            probs = logits.softmax(dim=-1)
            # compute uniform probs
            uniform = torch.ones_like(probs) / probs.shape[-1]
            # mix the NN probs with the uniform probs
            probs = (1 - unimix) * probs + unimix * uniform
            # compute the new logits
            logits = probs_to_logits(probs)
        return logits

    def update(self, buffer, step):
        """
              Trains the world model by using the transition buffer. Returns losses as defined in the Dreamer v3 paper.
        """
        # Sample transitions, containing batches of sequences of observation, actions, next_observation tuples from transition buffer.
        batch_size = self.batch_size
        trajectory_len = self.unroll_steps
        free_bits= self.free_bits 
        
        
        with torch.no_grad():
            transitions = buffer.sample(batch_size=batch_size, trajectory_len=trajectory_len+1, to_device=self.device, keys=["observation","action","state"])
            
            if transitions is None:
                return {}

            observations, actions, next_observations, states = transitions["observation"][:,:trajectory_len], transitions["action"][:,:trajectory_len], transitions["observation"][:,1:], transitions["state"][:,:trajectory_len]

            states = RSSMState.from_data(states, self.DET, self.LAT, self.CAT)
            if self.update_initial_state == "zero":
            #Initialize RSSM model with zero-state (i.e. no prior knowledge)
                prev_state = self.rssm.get_initial_state(batch_size=batch_size)
            elif self.update_initial_state  == "real": # use current (real world) state
                prev_state = self.state.repeat((batch_size, 1)).detach()

            elif self.update_initial_state  == "observed":
                prev_state = states[:,0,:] # initial state of trajectory
            else:
                raise NotImplementedError
            
        total_loss = torch.zeros((1,), device=self.device)

        # Rollout observations: "replaying" the transition while computing deterministic and stochastic hidden states
        prior_states, posterior_states = self.rssm.rollout_observation(seq_len=trajectory_len, observations=next_observations, actions=actions, prev_state=prev_state)  # type: ignore

        #### Representation & dynamics loss
        
        # Prevent zeros in logits
        new_logits_prior = torch.max(posterior_states.logits, torch.ones_like(posterior_states.logits)*1e-8)
        #new_logits_prior = self.uniform_mix(new_logits_prior, unimix=self.unimix)
        posterior_states = RSSMState(deter=posterior_states.deter, stoch=posterior_states.stoch, logits=new_logits_prior)
        
        new_logits_posterior = torch.max(prior_states.logits, torch.ones_like(prior_states.logits)*1e-8)
        #new_logits_posterior = self.uniform_mix(new_logits_posterior, unimix=self.unimix)
        prior_states = RSSMState(deter=prior_states.deter, stoch=prior_states.stoch, logits=new_logits_posterior)

        representation_loss_raw = torch.distributions.kl.kl_divergence(posterior_states.distribution(unimix=self.unimix), prior_states.detach().distribution(unimix=self.unimix)).reshape((batch_size, trajectory_len)) # type: ignore
        dynamics_loss_raw = torch.distributions.kl.kl_divergence(posterior_states.detach().distribution(unimix=self.unimix), prior_states.distribution(unimix=self.unimix)).reshape((batch_size, trajectory_len)) # type: ignore

        # Free bits
        representation_loss = torch.max(representation_loss_raw, free_bits*torch.ones_like(representation_loss_raw)).mean()
        dynamics_loss = torch.max(dynamics_loss_raw, free_bits*torch.ones_like(dynamics_loss_raw)).mean()

        total_loss += self.REP_SCALE * representation_loss
        total_loss += self.DYN_SCALE * dynamics_loss
        
        
        # Convert batched sequence to flattened sequence of size batch_size*sequence_len

        prior_states = prior_states.reshape((-1, prior_states.shape[-1])) # type: ignore
        posterior_states = posterior_states.reshape((-1, posterior_states.shape[-1])) # type: ignore
        
        return_dict = {"dyn"  : dynamics_loss.detach() , "rep" : representation_loss.detach(), "dyn_raw"  : dynamics_loss_raw.mean().detach() , "rep_raw" : representation_loss_raw.mean().detach()}


        # Reconstruct observations from posteriors 
        # rec_loss = self.rssm.decoder.reconstruction_loss(posterior_states, next_observations.reshape((batch_size*trajectory_len, *next_observations.shape[2:])))
        # return_dict["rec_loss"] = rec_loss.detach()

        # Reconstruction loss
        if False:
            reconstructed_observation = self.rssm.decoder(posterior_states.combined) 
            reconstruction_loss = RMSELoss()(reconstructed_observation.reshape((batch_size*trajectory_len, *next_observations.shape[2:])), next_observations.reshape((batch_size*trajectory_len, *next_observations.shape[2:])).detach())
        else:
            rec_loss, rec_dict = self.rssm.reconstruction_loss(posterior_states.reshape((batch_size, trajectory_len, -1)), next_observations.reshape((batch_size, trajectory_len, *next_observations.shape[2:])))

        total_loss += rec_loss
        return_dict = return_dict | rec_dict

        if total_loss.requires_grad:
            self.opt.zero_grad(set_to_none=True)
            total_loss.backward()
            self.opt.step()
        return return_dict
    