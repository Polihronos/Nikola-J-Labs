import numpy as np

class SimpleBayes:
  def __init__(self, hypotheses, decay_factor=0.2):
    self.hypotheses = hypotheses
    # Number of hypotheses
    self.n = len(hypotheses)

    self.decay = decay_factor

    # Uniform prior belief. 
    # Makes np array with 1's, the size of hypotheses and divides it by the size of hypotheses. 
    # The sum of all elements equals to 1
    self.prior = np.ones(self.n) / self.n

  def update(self, likelihoods):
    # likelihoods: A list of probabilities summing to 1.
    # Posterior = (Likelihood * Prior) / Normalization
    unnormalized_posterior = likelihoods * self.prior
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    # new belief gets the remaining mass ( 1 - self.decay) 
    # old belief gets decay -            (self.decay)
    self.prior = (1 - self.decay) * posterior + (self.decay * self.prior)

    '''
     Pure Bayes would be: 
     self.prior = posterior
    '''

    return posterior 



  
# Starting the class
model = SimpleBayes(["A", "B"], decay_factor=0.2)
print(f"Start Prior: {model.prior}") 
print("-" * 30)

# Evidence 1
evidence_1 = np.array([0.8, 0.2]) 
posterior_1 = model.update(evidence_1)
print(f"STEP 1 INPUT: {evidence_1}")
print(f"Result (Posterior): {posterior_1}")
print(f"Saved for Next (New Prior): {model.prior}")
print("-" * 30)

# Evidence 2
evidence_2 = np.array([0.4, 0.6])
posterior_2 = model.update(evidence_2)
print(f"STEP 2 INPUT: {evidence_2}")
print(f"Result (Posterior): {posterior_2}")
print(f"Saved for Next (New Prior): {model.prior}")