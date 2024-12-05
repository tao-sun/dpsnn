import numpy as np

from pesq import pesq_batch, PesqError

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from ..layers.spike_neuron import act_fun_adp
from ..layers.sdr import singlesrc_neg_sisdr
from ..layers.spike_neurons import get_neuro
from ..layers.srnn import SRNN, ReadoutLayer


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2).contiguous()
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2).contiguous()
        return x


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x
    
    def get_output_size(self, input_size):
        """
        Output = ((I-K+2P)/S + 1), where
        I - a size of input neuron
        K - kernel size,
        P - padding,
        S - stride
        """

        # Conv
        curr_size = curr_size - self.kernel_size + 1 # o = i + p - k + 1
        assert (curr_size > 0)

        # Strided conv/decimation
        assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
        curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size


class BinaryConv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.threshold = 0
        init_threshold = 0.0
        self.threshold = torch.nn.Parameter(torch.tensor(init_threshold, dtype=torch.float))

    def forward(self, x):
        x = super().forward(x)
        # x = torch.where(x > self.threshold, x, 0.0)
        x = act_fun_adp(x - self.threshold)
        # x = atan.apply(x - self.threshold, 2.0)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class SpikeConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, context_step, lif_type, lif_config):
        super(SpikeConv1d, self).__init__()

        self.context_step = context_step
        # depthwise conv
        self.dconv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=context_step+1,
            stride=context_step+1,
            groups=input_channels,
            bias=True)
        
        # self.norm = ChannelWiseLayerNorm(conv_channels, elementwise_affine=True)
        self.neuron = get_neuro(lif_type, **lif_config)

    def forward(self, y, time_step):
        y = self.dconv(y)
        # print(f"y_step shape: {y_step.shape}")
        # y = self.norm(y)
        y, _ = self.neuron(y, time_step)
        # y = torch.where(y > 0, 1.0, 0.0)
        return y


class StreamSpikeNet(pl.LightningModule):
    plif_config = {"init_tau":2.0}
    alif_config = {"tau_initializer": "multi_normal",
                "tau_m": [15, 20], "tau_m_initial_std": [5, 5],
                "tau_adp_initial": [200], "tau_adp_initial_std": [50]}
    
    def __init__(self, input_dim, context_dim, sr=16000,
                 L=20, stride=10, 
                 N=512, B=256, H=256, X=1,
                 learning_rate=1e-5, T_max=64):
        super().__init__()

        self.save_hyperparameters()
        
        print(f"input_dim: {input_dim}")
        print(f"context_dim: {context_dim}")
        self.context_dim = context_dim
        assert(context_dim % stride == 0)
        self.context_step = context_dim // stride
        rank_zero_info(f"context_step: {self.context_step}")
       
        self.sr = sr
        self.L = L
        self.B = B  # bottleneck
        self.N = N
        self.H = H  # channels in scnn and srnn
        assert(H % B == 0)
        self.X = X
        self.stride = stride if stride else  L // 2
        
        self.feature_steps = (input_dim - L) // (stride) + 1
        self.time_steps = self.feature_steps - self.X * self.context_step
        rank_zero_info(f"time steps: {self.time_steps}")

        self.encoder_1d = Conv1D(1, N, L, stride=self.stride, padding=0)
        self.encoder_act = nn.ReLU()
        self.ln = ChannelWiseLayerNorm(N)
        self.proj = BinaryConv1D(N, B, 1)

        blocks = []
        for _ in range(self.X):
            sconv1d = SpikeConv1d(B, H, self.context_step, "plif", self.plif_config)
            srnn = SRNN(H, B, "alif", self.alif_config)
            blocks.append(nn.ModuleList([sconv1d, srnn]))
        self.repeats = nn.ModuleList(blocks)
        
        self.srnn_readout = ReadoutLayer(self.X*B, B, "alif", self.alif_config)
        self.readout_threshold = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        
        self.mask = nn.Conv1d(B, N, 1)
        self.mask_act = nn.Sigmoid()  # could replaced by torch.nn.ReLU()
        
        self.decoder_1d = ConvTrans1D(N, 1, kernel_size=L, stride=self.stride, bias=True)

        self.lr = learning_rate
        self.T_max = T_max

        self.ops = self.time_step_ops()

    def forward(self, batch):
        inputs, targets = batch
        file_id, noisy_x, audio_len = inputs  # (b, seq)
        
        # Todo: add context x as padded 0
        context_wins = [None] * self.X
        enhanced_steps = []
        encoder_event_counts, proj_event_counts, readout_event_counts = [], [], []
        block_event_counts = [[[], []] for _ in range(self.X)]

        for feature_step in range(self.feature_steps):
            input_x = noisy_x[:, feature_step*self.stride:feature_step*self.stride+self.L]  # (b, L)

            # encoder
            x = self.encoder_1d(input_x)  # (b, N, 1)
            x = self.encoder_act(x)  # (b, N, 1)
            count = (torch.abs(x) > 0).to(x.dtype)
            encoder_event_counts.append(count)
            w = x  # (b, N, 1)

            # separator
            x = self.ln(x)  # (b, N, 1)
            proj_events = x = self.proj(x)  # (b, B, 1)
            count = (torch.abs(x) > 0).to(x.dtype)
            proj_event_counts.append(count)
            block_xs = []

            for block_idx in range(self.X):
                # accumulate contexts
                if feature_step >= block_idx * self.context_step and feature_step < (block_idx + 1) * self.context_step:
                    if  context_wins[block_idx] is None:
                        context_wins[block_idx] = x  # (b, B, 1)
                    else:
                        context_wins[block_idx] = torch.concat((context_wins[block_idx], x), dim=2)

                # run modules in self.X
                if feature_step >= (block_idx + 1) * self.context_step:
                    local_time_step = feature_step - (block_idx + 1) * self.context_step
                    block = self.repeats[block_idx]

                    # separator: SCNN
                    sconv1d = block[0]
                    win_w = torch.concat((context_wins[block_idx], x), dim=2)  # (b, B, context_step+1)
                    x = sconv1d(win_w, local_time_step)  # (b, H, 1)
                    count = (torch.abs(x) > 0).to(x.dtype)
                    block_event_counts[block_idx][0].append(count)

                    context_wins[block_idx]  = win_w[:, :, 1:]

                    # separator: SRNN
                    srnn = block[1]
                    x = srnn(x, local_time_step)  # (b, B)
                    x = torch.unsqueeze(x, dim=2)  # (b, B, 1)
                    count = (torch.abs(x) > 0).to(x.dtype)
                    block_event_counts[block_idx][1].append(count)

                    block_xs.append(x)
            
            # run remained modules
            if feature_step >= self.X * self.context_step:
                x = torch.concat(block_xs, dim=1)  # (b, X*B, 1)
                x = torch.squeeze(x, dim=2)  # (b, X*B)

                local_time_step = feature_step - self.X * self.context_step
                x = self.srnn_readout(x, local_time_step)  # (b, B)
                readout_events = x = torch.where(x > self.readout_threshold, x, 0.0)  # (b, B)
                count = (torch.abs(x) > 0).to(x.dtype)
                readout_event_counts.append(count)
                x = torch.unsqueeze(x, dim=2)  # (b, B, 1)
                
                # separator: masking
                x = self.mask(x)  # (b, N, 1)
                x = self.mask_act(x)  # (b, N, 1)
                x = w * x  # (b, N, 1)

                # decoder
                x = self.decoder_1d(x)  # (b, 1, L)
                enhanced_step = torch.squeeze(x, dim=1)  # (b, L)
                enhanced_steps.append(enhanced_step)
        enhanced_x = torch.stack(enhanced_steps, dim=2)  # (b, L, time_steps)
        output_size = (self.time_steps - 1) * self.stride + self.L
        # overlap-add in deconv
        fold = nn.Fold((1, output_size), kernel_size=(1, self.L), stride=(1, self.stride))
        enhanced_x = torch.squeeze(fold(enhanced_x))  # (b, output_size)

        event_rates = []
        # encoder_rate = torch.mean(torch.concat(encoder_event_counts))
        # event_rates.append(encoder_rate)
        proj_rate = torch.mean(torch.concat(proj_event_counts))
        event_rates.append(proj_rate)
        for block_idx in range(self.X):
            block_counts = block_event_counts[block_idx]
            for module_counts in block_counts:
                rate = torch.mean(torch.concat(module_counts))
                event_rates.append(rate)
        readout_rate = torch.mean(torch.concat(readout_event_counts))
        event_rates.append(readout_rate)
        return enhanced_x, torch.tensor(event_rates), proj_events, readout_events
    
    def common_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        out, event_rates, proj_events, readout_events = self(batch)
        return out, event_rates, proj_events, readout_events

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        _, y, _ = targets
        batch_size = y.shape[0]
        out, _, proj_events, readout_events = self.common_step(batch, batch_idx)
        y = torch.squeeze(y, dim=1)[:, self.context_dim*self.X:]
        # print(f"out shape: {out.shape}, y shape: {y.shape}")
        sisnr_loss = singlesrc_neg_sisdr(out, y).mean()
        mse_loss = nn.MSELoss()(out, y)
        proj_loss = torch.norm(proj_events, 1) / np.prod(proj_events.shape)
        readout_loss = torch.norm(readout_events, 1) / np.prod(readout_events.shape)
        # print(f"sisnr_loss: {sisnr_loss}, mse_loss: {mse_loss}")

        loss = 0.001 * mse_loss + 100 + sisnr_loss + 0.001 * proj_loss + 0.001 * readout_loss
        # loss = 0.001 * mse_loss + 100 + sisnr_loss
        
        gradient_norm = self.compute_gradient_norm()

        # Logging to TensorBoard by default
        self.log("loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("gn", gradient_norm, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("mse", mse_loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("proj", proj_loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("read", readout_loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("sisnr", sisnr_loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        # self.log("mse_loss", mse_loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, sync_dist=True)

        if batch_idx % 500 == 0:
            
            # sisnr = singlesrc_neg_sisdr(out, y).mean()
            rank_zero_info(f"sisnr: {sisnr_loss}")
        #     y = y[:, self.context_dim:y.shape[-1]-self.context_dim]
        #     targets = torch.squeeze(y).view(batch_size, -1)
        #     pesq_scores = self._compute_pesq(out[0].detach().cpu().numpy(), 
        #                                      targets.cpu().numpy(), 
        #                                      np.array([targets.shape[1]]*batch_size), 
        #                                      self.sr)
        #     rank_zero_info(f"pesq: {pesq_scores[:3]}")
        #     train_accu = self.train_accuracy(torch.sum(class_spikes[self.start_timestep:], 0), y)
            # train_accu = self.train_accuracy(torch.mean(membranes[self.start_timestep:], 0), y)
            # print(f"train accuracy: {train_accu}")

        # self.log("accu", train_accu, on_step=True, prog_bar=True)
        # torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        _, y, _ = targets
        batch_size = y.shape[0]
        out, _, _, _ = self.common_step(batch, batch_idx)
        y = torch.squeeze(y, dim=1)[:, self.context_dim*self.X:]
        sisnr_loss = singlesrc_neg_sisdr(out, y).mean()
        mse_loss = nn.MSELoss()(out, y)
        
        loss = 0.001 * mse_loss + 100 + sisnr_loss
        # loss = 0.001 * mse_loss + 100 + sisnr_loss + 0.01 * proj_loss + 0.001 * readout_loss

        # loss = sisnr_loss
        # try:
        #     pesq_val = torch.mean(eval_pesq(out, y, self.sr, 'wb'))
        # except Exception:
        #     pesq_val = torch.tensor(0.0)

        # Logging to TensorBoard by default
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_sisnr", sisnr_loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        # self.log("val_pesq", pesq_val, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        _, _, audio_length = inputs
        _, y, _ = targets
        batch_size = y.shape[0]

        out, event_rates, _, _ = self.common_step(batch, batch_idx)
        # y = torch.squeeze(y, dim=1)[:, self.context_dim*self.X:y.shape[-1]-self.delay_dim]
        # loss = singlesrc_neg_sisdr(out, y).mean()

        # batch_size = y.shape[0]
        # pesq_scores = self._compute_pesq(out[0].cpu().numpy(), 
        #                                  targets.cpu().numpy(), 
        #                                  np.array([targets.shape[1]]*batch_size), 
        #                                  self.sr)
        # self.log("val_pesq", np.mean(pesq_scores), on_epoch=True, prog_bar=True, sync_dist=True)

        # Logging to TensorBoard by default
        
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return out, event_rates
    
    def time_step_ops(self):
        encoder_ops = np.prod(self.encoder_1d.weight.shape)
        layernorm_ops = 6 * self.N + 5
        proj_ops = np.prod(self.proj.weight.shape)
        # sconv1d_ops = np.prod(self.repeats[0][0].dconv.weight.shape)
        mask_ops = np.prod(self.mask.weight.shape) if hasattr(self, 'mask') else 0
        sigmoid_ops = 4 * self.N
        decoder_ops = np.prod(self.decoder_1d.weight.shape)

        ops = {}
        ops["fixed_syn_ops"] = encoder_ops + layernorm_ops + proj_ops + sigmoid_ops + decoder_ops 
        ops["fixed_sep_syn_ops"] = layernorm_ops + proj_ops + sigmoid_ops
        
        ops["event_syn_ops"] = []
        for block_idx in range(self.X):
            block = self.repeats[block_idx]

            scnn_syn_ops = np.prod(block[0].dconv.weight.shape)
            ops["event_syn_ops"].append(scnn_syn_ops)
            srnn_syn_ops = np.prod(block[1].dense.weight.shape) + np.prod(block[1].recurrent.weight.shape)
            ops["event_syn_ops"].append(srnn_syn_ops)
            
        srnn_readout_syn_ops = np.prod(self.srnn_readout.dense.weight.shape)
        ops["event_syn_ops"].append(srnn_readout_syn_ops)
        ops["event_syn_ops"].append(mask_ops)
        
        ops['neuron_ops'] = self.X * (self.H + self.B)
        if hasattr(self, 'mask'):
            ops['neuron_ops'] = ops['neuron_ops'] + self.B
        else:
            ops['neuron_ops'] = ops['neuron_ops'] + self.N

        return ops
    
    def compute_gradient_norm(self, norm_type: float=2.0):
        parameters = self.parameters()
        if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return total_norm

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         # REQUIRED: The scheduler instance
        #         "scheduler": lr_scheduler,
        #         "interval": "epoch",
        #         "frequency": 1
        #     }
        # }
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        stepping_batches = self.trainer.estimated_stepping_batches

        # set min_lr
        min_lr = 1e-3
        estimated_final_lr = self.lr * (0.9999 ** stepping_batches)
        if estimated_final_lr >= min_lr:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, (min_lr/self.lr) ** (1/stepping_batches))

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step'  # called after each training step
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler  
        }
    
    def _compute_pesq(self, pred_wavs, target_wavs, lengths, sr):
        # pesq_scores = Parallel(n_jobs=30)(
        #     delayed(pesq)(
        #         fs=sr,
        #         ref=clean[: int(length)],
        #         deg=enhanced[: int(length)],
        #         mode="wb",
        #         on_error=PesqError.RETURN_VALUES
        #     )
        #     for enhanced, clean, length in zip(pred_wavs, target_wavs, lengths)
        # )
        pesq_scores = pesq_batch(fs=sr, 
                                 ref=target_wavs, 
                                 deg=pred_wavs, 
                                 mode="wb", 
                                 on_error=PesqError.RETURN_VALUES)
        pesq_scores = [score for score in pesq_scores if score != -1]
        return pesq_scores
