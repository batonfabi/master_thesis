""" 
inspired by  https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/train.py
MIT License

Copyright (c) 2020 Aladdin Persson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. """

import random
import os
import torch
import torchvision
import math
from utils import gradient_penalty
from fid_score import FIDscore
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances 
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer

class WGAN_GP_training():
    """
    Implementation of wgan-gp training
    """

    # params {critic_iterations:5, latent_space_dim:100, lambda_gp: 2e-4, experiment_folder: 'exp1', log_iterations:100, calc_fid:true, evaluation_samples: 1000, device:'cuda'}
    def __init__(self, generator, critic, train_generator, opt_gen, opt_critic, params):
        self.gen = generator.to(params["device"])
        self.critic = critic.to(params["device"])
        self.train_generator = train_generator
        self.opt_gen = opt_gen 
        self.opt_critic = opt_critic 
        self.critic_iterations = params["critic_iterations"]
        self.latent_space_dim = params["latent_space_dim"]
        self.lambda_gp = params["lambda_gp"]
        self.experiment_folder = params["experiment_folder"]
        self.evaluation_samples = params["evaluation_samples"]
        self.log_iterations = params["log_iterations"]
        self.device = params["device"]
        self.fixed_noise = torch.randn(32, params["latent_space_dim"], 1, 1).to(params["device"])
        self.writer_real = SummaryWriter("logs/" + self.experiment_folder + "/real")
        self.writer_fake = SummaryWriter("logs/" + self.experiment_folder + "/fake")
        self.writer_loss = SummaryWriter("logs/" + self.experiment_folder + "/loss")
        self.writer_gp = SummaryWriter("logs/" + self.experiment_folder + "/gp")
        self.writer_fid = SummaryWriter("logs/" + self.experiment_folder + "/fid")
        self.writer_euc = SummaryWriter("logs/" + self.experiment_folder + "/euc")
        self.save_dir = "logs/" + self.experiment_folder + "/sav/"
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.epochs = 0
        self.steps = 0
        self.epoch_len = len(train_generator)
        self.logged_steps = 0
        self.calc_fid = params["calc_fid"]
        if self.calc_fid:
            self.fid_calculator = FIDscore()
            self.euc_truth = self._avg_euc_dataset(self.train_generator, 10, 1000)
            self.fid_truth = self._avg_fid_dataset(self.train_generator, 20)
            print( f"Initialized ground truth. Eucledian truth: {self.euc_truth:.4f}, fid truth: {self.fid_truth:.4f}")
        
    def get_models(self):
        return self.gen, self.critic

    def train_epoch(self):
        self.epochs = self.epochs + 1 
        self.gen.train()
        self.critic.train()
        for batch_idx, (real, _) in enumerate(self.train_generator):
            self.steps = self.steps + 1   
            batch_data = real.to(self.device)
            loss_gen, loss_critic, last_fake_batch = self._train_epoch(batch_data)
            self.writer_loss.add_scalars('Loss/train', {'critic': loss_critic, 'gen':loss_gen}, self.steps)
            if batch_idx % self.log_iterations == 0: #and batch_idx > 0:
                self._sample_images(batch_data)
                if batch_idx % (self.log_iterations * 3) == 0 and batch_idx > 0:
                    torch.save(self.gen, self.save_dir + "gen" + str(self.steps) + ".torch")
                torch.save(self.critic, self.save_dir + "crit" + str(self.steps) + ".torch")
                if self.calc_fid:
                    fid_score = self._avg_fid_generator(self.gen, self.train_generator, self.evaluation_samples)
                    euc_score = self._avg_euc_generator(self.gen, 1000)
                    self.writer_fid.add_scalars("fid", {"fid_gan": fid_score, "ground_truth": self.fid_truth}, self.steps)   
                    self.writer_euc.add_scalars("euc", {"euc_gan": euc_score, "ground_truth": self.euc_truth}, self.steps)   
                    self._log_progress(batch_idx, loss_gen, loss_critic, fid_score, self.fid_truth, euc_score, self.euc_truth)
                else:
                    self._log_progress(batch_idx, loss_gen, loss_critic)
                    

        self.gen.eval()
        self.critic.eval()

        torch.save(self.gen, self.save_dir + "gen_epoch" + str(self.epochs)+ ".torch")
        torch.save(self.critic, self.save_dir + "crit_epoch" + str(self.epochs)+ ".torch")

    def _avg_fid_generator(self, gan_generator, data_generator, samples):
        gan_generator.eval()
        calc_samps = samples
        with torch.no_grad():
            sum = 0
            for x in range(samples):
                np_real = data_generator.__getitem__(x)[0].cpu().numpy()
                batch_size = np.shape(np_real)[0]
                np_real = torch.Tensor(np.concatenate((np_real,np_real,np_real), axis=1)).to(self.device)

                random_vecors = torch.randn(batch_size, self.latent_space_dim, 1, 1).to(self.device)
                np_fake = gan_generator(random_vecors).cpu().numpy()
                np_fake = torch.Tensor(np.concatenate((np_fake,np_fake,np_fake), axis=1)).to(self.device)
                try:
                    sum_tmp = self.fid_calculator.calculate_fretchet(np_real, np_fake)
                    sum += sum_tmp
                except: 
                    calc_samps = calc_samps - 1
                    
        gan_generator.train()
        return sum/calc_samps

    def _avg_fid_dataset(self, data_generator, samples):
        start = timer()
        sum = 0
        run_count = 0 
        run = True 
        prev = None
        while run:
            for _, (real, _) in enumerate(data_generator):
                np_real = real.cpu().numpy()
                np_real = torch.Tensor(np.concatenate((np_real,np_real,np_real), axis=1)).to(self.device)
                if prev == None:
                    prev = np_real
                else:
                    run_count += 1 
                    sum += self.fid_calculator.calculate_fretchet(np_real, prev)
                    prev = np_real
                    if run_count == samples:
                        run = False
                        break
                    
        end = timer()
        return sum/samples

    def _avg_euc_dataset(self, data_generator, epochs, samples):
        start = timer()
        sum = 0
        t = []
        for _ in range(epochs):
            for batch_idx, (real, _) in enumerate(data_generator):
                t.append(real)
                if batch_idx == math.ceil(samples/len(real)):
                    break
            euc = self._calc_euclidean(torch.Tensor(np.concatenate(t, axis=0)))
            sum += euc 
            data_generator.shuffle_dataset()
            t = []
                    
        end = timer()
        return sum/epochs

    def _avg_euc_generator(self, gan_generator, samples):
        gan_generator.eval()
        with torch.no_grad():
            random_vecors = torch.randn(samples, self.latent_space_dim, 1, 1).to(self.device)
            batch = gan_generator(random_vecors).cpu().numpy()
        gan_generator.train()           
        euc = self._calc_euclidean(batch)
        return euc

    def _calc_euclidean(self, batch):
        batch_shape = np.shape(batch)
        reshaped_batch = np.reshape(batch, (batch_shape[0], batch_shape[2] * batch_shape[3]))
        euc = euclidean_distances(reshaped_batch)
        identity = np.identity(batch_shape[0]) * np.max(euc)
        euc = euc + identity
        euc_score = np.average(euc.min(axis=1))
        return euc_score

    def _train_epoch(self, real_data):
        cur_batch_size = real_data.shape[0]
        for _ in range(self.critic_iterations):
            noise = torch.randn(cur_batch_size, self.latent_space_dim, 1, 1).to(self.device)
            fake = self.gen(noise)
            critic_real = self.critic(real_data).reshape(-1)
            critic_fake = self.critic(fake).reshape(-1)
            gp = gradient_penalty(self.critic, real_data, fake, device=self.device)
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp* gp)
            self.critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.opt_critic.step()

        self.writer_gp.add_scalar("gp", gp.cpu().detach().numpy(), self.steps)   
        gen_fake = self.critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        self.gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

        
        return loss_gen, loss_critic, fake 

    def _sample_images(self, real_data):
        with torch.no_grad():
            fake = self.gen(self.fixed_noise)
            img_grid_real = torchvision.utils.make_grid(real_data[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

            self.writer_real.add_image("Real", img_grid_real, global_step=self.logged_steps)
            self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.logged_steps)

        self.logged_steps = self.logged_steps + 1
    
    def _log_progress(self, batch_idx, loss_gen, loss_critic, fid=-1, fid_truth =-1, euc=-1, euc_truth=-1):
        print(
            f"Epoch [{self.epochs}] Batch {batch_idx}/{self.epoch_len} \
              Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}, euc: {euc:.4f}, euc_truth: {euc_truth:.4f}, fid: {fid:.4f}, fid_truth: {fid_truth:.4f}"
        )






