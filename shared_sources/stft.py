"""
This methods and classes are based on https://github.com/NVIDIA/tacotron2/blob/master/stft.py 
and are used to transofm the audio samples to mel spectograms for the pretrained vocoder

BSD 3-Clause License
Copyright (c) 2017, Prem Seetharaman
All rights reserved.
* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
from librosa.filters import mel as librosa_mel_fn
import torch
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from torch.autograd import Variable
import torch.nn.functional as F

class CalcMel():
    def __init__(self, scaler, output_format = (40, 1, 80, 80), filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80,
     sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0, window='hann'):
        self.stft_fn = STFT(filter_length,hop_length,win_length)
        self.mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        self.mel_basis = torch.from_numpy(self.mel_basis).float()
        self.scaler = scaler
        self.output_format = output_format

    def dynamic_range_compression(self, x, C=1, clip_val=1e-5):
        """
        PARAMS
        ------
        C: compression factor
        """
        x = torch.Tensor(x.float())
        return torch.log(torch.clamp(x, min=clip_val) * C).numpy()
    
    def spectral_normalize(self, magnitudes):
        output = self.dynamic_range_compression(magnitudes)
        return output
    
    def spectral_de_normalize(self, magnitudes):
        output = self.dynamic_range_decompression(magnitudes)
        return output
    
    def mel_spectrogram(self, y, rescale=True):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        y = torch.tensor(y)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        if rescale:
            for i in range(len(mel_output)):
                mel_output[i] = self.scaler.transform(mel_output[i])
        mel_output = np.reshape(mel_output, self.output_format)
        return mel_output

class STFT():
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
        
        self.forward_basis = forward_basis.float()
        self.inverse_basis = inverse_basis.float()
        
    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), 
                                     stride=self.hop_length, padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase