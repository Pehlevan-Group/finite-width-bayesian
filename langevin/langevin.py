import torch
import numpy as np

dtype = torch.double
cpu = torch.device("cpu")
gpu = torch.device("cuda")

def model(depth, n0, nd, hidden_width, std = 1):
  if isinstance(hidden_width, list) is False:
    hidden_width = [hidden_width for i in range(depth-1)]
  else:
    assert(len(hidden_width) == depth - 1)
  
  widths = np.array([n0, *hidden_width, nd])

  W = []
  for i in range(len(widths) - 1):
    W += [torch.normal(0,std,(widths[i],widths[i+1]), device=gpu, dtype=dtype, requires_grad=True)]

  return W

class Langevin():
    def __init__(self, w_target, x_train, n0, nd, N_tr, N_test, depth, hidden_width, std = 1):
      ## Specify the dataset
      self.n0 = n0   # Input dimensionality
      self.nd = nd   # Output dimensionality
      self.std = std # Weight variance

      self.N_tr = N_tr     # Training set size
      self.N_test = N_test # Test set size

      self.w_target = w_target # Target function
      self.x_train = x_train

      ### Intialize the model and create random weight tensors
      self.depth = depth
      self.hidden_width = hidden_width
      self.W = model(self.depth, self.n0, self.nd, self.hidden_width, self.std)

      ## Store train and test errors
      self.train_error = []
      self.test_error = []

      ## Store the number of averaging
      self.avg_count = 0

      # Store weight averages
      self.W_avg = []
      for w in self.W:
        self.W_avg += [torch.zeros(w.shape,device=gpu,dtype=dtype)]

      # Store layer kernel averages
      self.K_avg = []
      for w in self.W:
        self.K_avg += [torch.zeros((N_tr,N_tr),device=gpu,dtype=dtype)]

      ## Store theory predictions
      self.Gxx = 0
      self.Gyy = 0
      self.Kl_theory_pert = []
      self.Kl_theory_exact = []

      return
    
    def train(self, dt, beta, nT, compute_test = False):
      ##### Experiment Parameters ######
      dt = dt      # Learning rate / timestep 
      beta = beta  # Inverse temperature
      nT = nT      # Number of training timesteps

      target = lambda x: x @ self.w_target.T
      y_train = target(self.x_train)

      # Iterate over training steps
      t_init = time.time()
      for t in range(nT):
        # ### Sample fresh inputs at each iteration
        # self.x_train = torch.normal(0,1,size=(self.N_tr, self.n0),device=gpu,dtype=dtype)/np.sqrt(self.n0)
        # y_train = self.target(self.x_train)

        # Manually compute the forward pass and loss, computing the activations at each layer
        h = []
        y_pred = self.x_train
        for w in self.W:
          y_pred = torch.matmul(y_pred, w) / np.sqrt(w.shape[0])
          h += [y_pred]
        loss = (y_pred - y_train).pow(2).sum()
        self.train_error += [loss.item()]

        if compute_test:
          # Compute the forward pass for the test error
          x_test = torch.normal(0,1,size=(self.N_test,self.n0),device=gpu,dtype=dtype)/np.sqrt(self.n0)
          y_test = self.target(x_test)
          yhat_pred = x_test
          for w in self.W:
            yhat_pred = torch.matmul(yhat_pred, w) / np.sqrt(w.shape[0])
          self.test_error += [(yhat_pred - y_test).pow(2).sum().detach().to(cpu).numpy()]
        else:
          self.test_error = [0]

        # Perform the backward pass
        loss.backward()

        # Manually handle weight updates
        with torch.no_grad():
            for w in self.W:
              # Perform one Euler-Maruyama step
              w -= dt * (w / beta + w.grad) + np.sqrt(2*dt/beta) * torch.randn(w.shape,device=gpu,dtype=dtype)
              # Manually zero the gradients after updating weights
              w.grad = None

        # Compute averaged weight matrices online once loss = noise
        if loss.item() < self.n0*self.nd/(2*beta) * 1.3 and t > int(nT / 2):
          self.avg_count += 1
          for i in range(len(h)):
            self.W_avg[i] += self.W[i].detach()
            self.K_avg[i] += (h[i] @ h[i].T).detach() / h[i].shape[1]

          # Print and store the training error
          if t % 1000 == 0 or t == 0: 
            print("Epoch: %d | loss: %.4e | test_err: %.4e | avg_count: %d | %0.2f sec."%(t, loss.item(), self.test_error[-1], self.avg_count, time.time()-t_init))

        if self.avg_count > 10000:
          break

      if self.avg_count == 0: print("Did not Converge!!!")
      print(f"Completed simulation in {time.time() - t_init} seconds.")

      # Move averages to CPU and convert to numpy
      W_avg_cpu = []
      for i, w_avg in enumerate(self.W_avg):
        W_avg_cpu += [w_avg.to(cpu).numpy()/self.avg_count]
      self.W_avg = W_avg_cpu

      K_avg_cpu = []
      for k_avg in self.K_avg:
        K_avg_cpu += [k_avg.to(cpu).numpy()/self.avg_count]
      self.K_avg = K_avg_cpu

      return

    def theory(self, beta):
      target = lambda x: x @ self.w_target.T

      self.Gxx = (self.x_train @ self.x_train.T)
      self.Gyy = (target(self.x_train) @ target(self.x_train).T/self.nd)

      I = torch.eye(self.N_tr, device=gpu, dtype=dtype)
      gamma = I/beta + self.Gxx
      gamma_inv = torch.linalg.inv(gamma)
      prefactor = self.nd/self.hidden_width

      self_energy = (self.Gyy - self.Gxx - I/beta) 

      self.Kl_theory_pert = []
      self.Kl_theory_exact = []
      
      for i in range(len(self.W)):
        self.Kl_theory_pert += [(self.Gxx @ (I + (i+1)*prefactor*gamma_inv @ self_energy @ gamma_inv @ self.Gxx)).cpu().numpy()]
        self.Kl_theory_exact += [(self.Gxx @ torch.linalg.inv(I - (i+1)*prefactor*gamma_inv @ self_energy @ gamma_inv @ self.Gxx)).cpu().numpy()]
      
      return

    def final_params():
      params = {'n0': self.n0, 'nd': self.nd, 'N_tr': self.N_tr, 'N_test': self.N_test,
                'w_target': self.w_target, 'x_train': self.x_train,
                'depth': self.depth, 'hidden_width': self.hidden_width,
                'avg_count': self.avg_count, 'W_avg': self.W_avg, 'K_avg': self.K_avg,
                'Gxx': self.Gxx, 'Gyy': self.Gyy, 
                'Kl_theory_pert': self.Kl_theory_pert,
                'Kl_theory_exact': self.Kl_theory_exact}

      return params

