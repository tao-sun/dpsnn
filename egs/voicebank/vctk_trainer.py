import time
import argparse, os
import random

from omegaconf import OmegaConf
from fvcore.nn import FlopCountAnalysis

import torch
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
import torchaudio

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as eval_pesq
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as eval_stoi

from dpsnn.data.wave_dataset2 import worker_init_fn
from dpsnn.data.wave_dataset2 import ContextSepDataset, EvaluationDataset
from dpsnn.data.augment import remix
from dpsnn.data.dnsmos import DNSMOS
from dpsnn.data.hdf5_prepare import create_hdf5
from dpsnn.data.voicebank_prepare import prepare_voicebank, download_vctk
from dpsnn.data.metrics import eval_composite
from dpsnn.layers.sdr import singlesrc_neg_sisdr

from dpsnn.models.dp_binary_net import StreamSpikeNet

def optimize_seeding():
    torch.backends.cudnn.benchmark = True
    _seed_ = 2020
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    random.seed(_seed_)

def randomize_seeding():
    torch.seed()  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = False
    np.random.seed()

optimize_seeding()


parser = argparse.ArgumentParser()
parser.add_argument('--random_seeds', action='store_true')
parser.add_argument('--config', type=str, required=False, default="vctk.yaml")
parser.add_argument('--test_ckpt_path', type=str, required=False)
parser.add_argument('--load_ckpt_path', type=str, required=False)
parser.add_argument('--frame_dur', type=float, required=False)
parser.add_argument('--context_dur', type=float, required=False)
parser.add_argument('--delay_dur', type=float, required=False)
parser.add_argument('-L', type=int, required=False, default=20)
parser.add_argument('--stride', type=int, required=False, default=10)
parser.add_argument('-N', type=int, required=False, default=256)
parser.add_argument('-H', type=int, required=False, default=256)
parser.add_argument('-B', type=int, required=False, default=256)
parser.add_argument('-K', type=int, required=False, default=100)
parser.add_argument('--alpha', type=float, required=False, default=0.5)
parser.add_argument('--beta', type=float, required=False, default=0.5)
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--lmbda', type=float, default=2.0, help='Lambda')
parser.add_argument('--liquid', action='store_true')
# parser.add_argument('-R', type=int, required=False, default=1)
parser.add_argument('-X', type=int, required=False, default=1)
parser.add_argument('--sr', type=int, required=False)
parser.add_argument('--bi_direction', action='store_true')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--neuro_type', type=str, required=False, default="plif")
parser.add_argument('--batch_size', type=int, required=False)
parser.add_argument('--max_epochs', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
parser.add_argument('--precision', type=str, required=False, help="16, 32, bf16")
parser.add_argument('--devices', nargs='+', type=int, default=[0])
parser.add_argument('--device_num', type=int, required=False)
parser.add_argument('--random_hops', action='store_false')


def rank_print(info):
    rank_zero_info(info)


args = parser.parse_args()
rank_print(args)

if args.random_seeds:
    randomize_seeding()

# This line will print the entire config of the LSTM model
# config_path = f"{args.config}"
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

conf_file_path = os.path.join(script_dir, args.config)
config = OmegaConf.load(conf_file_path)
config = OmegaConf.to_container(config, resolve=True)
config = OmegaConf.create(config)

if args.frame_dur:
    config.frame_dur = args.frame_dur
    rank_print(f"frame_dur: {config.frame_dur}")
if args.context_dur is not None:
    config.context_dur = args.context_dur
    rank_print(f"context_dur: {config.context_dur}")
if args.delay_dur is not None:
    config.delay_dur = args.delay_dur
    rank_print(f"delay_dur: {config.delay_dur}")
if args.sr:
    config.sample_rate = args.sr
    rank_print(f"sample_rate: {config.sample_rate}")

print("Trainer config - \n")
# accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
# config.trainer.accelerator = accelerator
rank_print(f"accelerator: {config.trainer.accelerator}")

if args.device_num:
    config.trainer.devices = args.device_num
    rank_print(f"devices: {config.trainer.devices}")
elif args.devices:
    config.trainer.devices = args.devices
    args.device_num = len(config.trainer.devices)
    rank_print(f"devices: {config.trainer.devices}")
if args.precision:
    if args.precision.isdecimal():
        args.precision = int(args.precision)
    config.trainer.precision = args.precision
    rank_print(f"precision: {config.trainer.precision}")
if args.max_epochs:
    config.trainer.max_epochs = args.max_epochs
    rank_print(f"max_epochs: {config.trainer.max_epochs}")
if args.lr:
    config.optim.lr = args.lr
    rank_print(f"learng_rate: {config.optim.lr}")
if args.batch_size:
    config.batch_size = args.batch_size
    # rank_print(f"batch_size: {config.batch_size}")

config.batch_size //=  args.device_num
rank_print(f"batch_size: {config.batch_size}")
# config.optim.lr *= len(config.trainer.devices)
# rank_print(f"learng_rate: {config.optim.lr}")



def normalize_estimates(est, mix):
    """Normalizes estimates according to the mixture maximum amplitude

    Args:
        est_np (np.array): Estimates with shape (n_src, time).
        mix_np (np.array): One mixture with shape (time, ).

    """
    # mix_max = torch.max(torch.abs(mix))
    # return est * mix_max / torch.max(torch.abs(est))
    return est / est.abs().max(dim=1, keepdim=True)[0]


def start_func():
    if not os.path.exists(config.data_folder):
        os.makedirs(config.data_folder)
        download_vctk(config.data_folder)
    prepare_voicebank(
        data_folder=config.data_folder,
        save_folder=config.save_folder
    )
    if not os.path.exists(config.hdf5_train):
        create_hdf5(config.csv_train, config.hdf5_train, config.sample_rate)
        create_hdf5(config.csv_valid, config.hdf5_valid, config.sample_rate)
        create_hdf5(config.csv_test, config.hdf5_test, config.sample_rate)

    # train_dataloader = make_dataloader(train=True,
    #                                data_kwargs={"hdf_file": config.hdf5_train},
    #                                batch_size=config.batch_size,
    #                                chunk_size=int(config.frame_dur * config.sample_rate),
    #                                num_workers=config.num_workers)
    # valid_dataloader = make_dataloader(train=True,
    #                                data_kwargs={"hdf_file": config.hdf5_test},
    #                                batch_size=config.batch_size,
    #                                chunk_size=int(config.frame_dur * config.sample_rate),
    #                                num_workers=config.num_workers)
    # test_dataloader = make_eval_dataloader(data_kwargs={"hdf_file": config.hdf5_test}, num_workers=config.num_workers)

    # train_dataset = WaveDataset(
    #     hdf_file=config.hdf5_train,
    #     frame_dur=config.frame_dur,
    #     sr=config.sample_rate,
    #     channels=1,
    #     context_dur=config.context_dur,
    #     max_frames=config.max_frames)
    # train_dataloader = DataLoader(train_dataset,
    #                         batch_size=config.batch_size,
    #                         shuffle=True,
    #                         collate_fn=frame_clip_batch,
    #                         num_workers=8,
    #                         worker_init_fn=worker_init_fn,
    #                         drop_last=True)
    
    start_context_dur = args.X * config.context_dur

    if args.test_ckpt_path is None:
        random_hops = False if args.augment else args.random_hops
        train_dataset = ContextSepDataset(
            hdf_file=config.hdf5_train,
            frame_dur=config.frame_dur,
            sr=config.sample_rate,
            channels=1,
            start_context_dur=start_context_dur,
            end_context_dur=config.delay_dur,
            random_hops=random_hops)
        def remix_collate(batch):
            inputs, targets = default_collate(batch)
            noisy, clean = inputs[1], targets[1]
            noisy_perm, clean = remix(noisy-clean, clean)
            inputs[1] = noisy_perm
            return inputs, targets
        
        collate_fn = remix_collate if args.augment else default_collate
        shuffle = False if args.augment else True
        train_dataloader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                shuffle=shuffle,
                                num_workers=8,
                                collate_fn=collate_fn,
                                worker_init_fn=worker_init_fn,
                                pin_memory=True)


        # valid_dataset = WaveDataset(
        #     hdf_file=config.hdf5_test,
        #     frame_dur=config.frame_dur,
        #     sr=config.sample_rate,
        #     channels=1,
        #     context_dur=config.context_dur,
        #     max_frames=config.max_frames)

        # valid_dataloader = DataLoader(valid_dataset,
        #                         batch_size=config.batch_size,
        #                         shuffle=False,
        #                         collate_fn=frame_clip_batch,
        #                         num_workers=8,
        #                         worker_init_fn=worker_init_fn,
        #                         drop_last=True)
        valid_dataset = ContextSepDataset(
            hdf_file=config.hdf5_test,
            frame_dur=config.frame_dur,
            sr=config.sample_rate,
            channels=1,
            start_context_dur=start_context_dur,
            end_context_dur=config.delay_dur,
            random_hops=False)
        valid_dataloader = DataLoader(valid_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=8,
                                worker_init_fn=worker_init_fn,
                                pin_memory=True)
    
    test_dataset = EvaluationDataset(
        hdf_file=config.hdf5_test,
        frame_dur=config.frame_dur,
        sr=config.sample_rate,
        channels=1,
        start_context_dur=start_context_dur,
        end_context_dur=config.delay_dur)

    test_dataloader = DataLoader(test_dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=8)


    if args.test_ckpt_path is None:
        train_shapes = train_dataset.get_shapes()
    else:
        train_shapes = test_dataset.get_shapes()
    print(f"shapes: {train_shapes}")
    input_dim = train_shapes["input_size"]
    output_dim = train_shapes["output_size"]
    assert(train_shapes["start_context_size"] % args.X == 0)
    context_dim = train_shapes["start_context_size"] // args.X
    delay_dim = train_shapes["end_context_size"]
    rank_print(f"context_dim: {context_dim}")


    class TestCallback(pl.Callback):
        dt = args.stride / config.sample_rate
        latency = args.L / config.sample_rate
        
        flops_init = False
        test_event_rates = []
        total_flops = 0

        noisy_sisnr_list, sisnr_list = [], []

        noisy_pesqs = np.zeros(1)
        clean_pesqs = np.zeros(1)
        enhanced_pesqs = np.zeros(1)

        noisy_stois = np.zeros(1)
        clean_stois = np.zeros(1)
        enhanced_stois = np.zeros(1)
        
        composite_noisy = np.zeros(4)
        composite_clean = np.zeros(4)
        composite_enhanced  = np.zeros(4)

        dnsmos = DNSMOS()
        dnsmos_noisy = np.zeros(3)
        dnsmos_clean = np.zeros(3)
        dnsmos_enhanced  = np.zeros(3)


        def on_test_epoch_end(self, trainer, pl_module):
            rank_print("\nTesting completed!\n")

            mean_event_rates = np.mean(self.test_event_rates, axis=0)
            syn_ops = []
            for rate, block_syn_ops in zip(mean_event_rates, self.event_syn_ops):
                syn_ops.append(rate * block_syn_ops)
            effective_synops_rate = (sum(syn_ops) + self.fixed_syn_ops + 10 * self.neuron_ops) / self.dt
            synops_delay_product = effective_synops_rate * self.latency
            sep_synops_rate = (sum(syn_ops) + self.fixed_sep_syn_ops + 10 * self.neuron_ops) / self.dt
            sep_synops_delay_product = sep_synops_rate * self.latency
            spike_synops_rate = (sum(syn_ops) + 10 * self.neuron_ops) / self.dt
            spike_synops_delay_product = spike_synops_rate * self.latency

            rank_print(f"total flops: {self.total_flops}")
            for module_idx, mean_event_rate in enumerate(mean_event_rates):
                rank_print(f"final event fire rate for module {module_idx}: {mean_event_rate}")
            rank_print(f'final power proxy (Effective SynOPS)   : {effective_synops_rate:.3f} ops/s')
            rank_print(f'final PDP proxy (SynOPS-delay product) : {synops_delay_product: .3f} ops')

            rank_print(f'Avg power proxy (Effective SynOPS, excluding encoder and decoder)   : {sep_synops_rate:.3f} ops/s')
            rank_print(f'Avg PDP proxy (SynOPS-delay product, excluding encoder and decoder) : {sep_synops_delay_product: .3f} ops')
            rank_print(f'final spike power proxy (Effective SynOPS)   : {spike_synops_rate:.3f} ops/s')
            rank_print(f'final spike PDP proxy (SynOPS-delay product) : {spike_synops_delay_product: .3f} ops')

            rank_print(f"final sisnr mean: {torch.mean(torch.tensor(self.sisnr_list))}, {torch.mean(torch.tensor(self.noisy_sisnr_list))}\n")
            
            noisy_pesqs = self.noisy_pesqs / len(self.noisy_sisnr_list)
            clean_pesqs = self.clean_pesqs / len(self.noisy_sisnr_list)
            enhanced_pesqs = self.enhanced_pesqs / len(self.noisy_sisnr_list)
            rank_print(f"\nfinal pesq [noisy, clean, enhanced]: {noisy_pesqs}, {clean_pesqs}, {enhanced_pesqs}")
            noisy_stois = self.noisy_stois / len(self.noisy_sisnr_list)
            clean_stois = self.clean_stois / len(self.noisy_sisnr_list)
            enhanced_stois = self.enhanced_stois / len(self.noisy_sisnr_list)
            rank_print(f"\nfinal stoi [noisy, clean, enhanced]: {noisy_stois}, {clean_stois}, {enhanced_stois}")
            
            self.composite_clean /= len(self.noisy_sisnr_list)
            self.composite_noisy /= len(self.noisy_sisnr_list)
            self.composite_enhanced /= len(self.noisy_sisnr_list)

            print('\nfinal composite clean   [pesq, ovrl, sig, bak]: ', self.composite_clean)
            print('final composite noisy   [pesq, ovrl, sig, bak]: ', self.composite_noisy)
            print('final composite enhanced   [pesq, ovrl, sig, bak]: ', self.composite_enhanced)

            self.dnsmos_clean /= len(self.sisnr_list)
            self.dnsmos_noisy /= len(self.sisnr_list)
            self.dnsmos_enhanced /= len(self.sisnr_list)

            rank_print(f"\nAvg DNSMOS clean   [ovrl, sig, bak]: {self.dnsmos_clean}")
            rank_print(f"Avg DNSMOS noisy   [ovrl, sig, bak]: {self.dnsmos_noisy}")
            rank_print(f"Avg DNSMOS enhanced [ovrl, sig, bak]: {self.dnsmos_enhanced}")
            
            # dt = args.stride / config.sample_rate
            # buffer_latency = args.L / config.sample_rate
            # print(f'Buffer latency: {buffer_latency * 1000} ms')

            # dns_latency = np.mean(self.dns_delays) / config.sample_rate
            # print(f'Network latency: {dns_latency * 1000} ms')

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            x, y = batch

            if not self.flops_init:
                self.total_flops = FlopCountAnalysis(pl_module, batch).total()
                
                self.fixed_syn_ops = pl_module.ops['fixed_syn_ops']
                self.fixed_sep_syn_ops = pl_module.ops['fixed_sep_syn_ops']
                self.event_syn_ops = pl_module.ops['event_syn_ops']
                self.neuron_ops = pl_module.ops['neuron_ops']

                self.flops_init = True
            
            noisy = x[1][:, context_dim*args.X:x[1].shape[-1]-delay_dim]
            clean = torch.unsqueeze(y[1], 0)
            enhanced, event_rates = outputs
            self.test_event_rates.append(event_rates.cpu().data.numpy())
            
            # print(f"noisy: {noisy.shape}")
            # print(f"clean: {clean.shape}")
            # print(f"enhanced: {enhanced.shape}")

            file_id = y[0]
            file_len = y[2]
            save_dir = os.path.join(config.output_folder, "")
            
            # orig_noisy_sisnr = -singlesrc_neg_sisdr(noisy, clean)
            # orig_enhanced_sisnr = -singlesrc_neg_sisdr(enhanced, clean)
            # # print(f"x[1] shape: {x[1].shape}, y[1] shape: {y[1].shape}")
            # # for idx in range(orig_noisy_sisnr.shape[0]):
            #     # print(f"{orig_enhanced_sisnr[idx]}, {orig_noisy_sisnr[idx]}")
            # # print(f"\n\nsegmental enhanced sisnr: {orig_enhanced_sisnr.mean()}, noisy_sisnr: {orig_noisy_sisnr.mean()}")
            # self.orig_noisy_sisnr_list.append(orig_noisy_sisnr.mean())
            # self.orig_sisnr_list.append(orig_enhanced_sisnr.mean())

            enhanced_tensor = enhanced.reshape(1, -1)
            # print(f"enhanced_tensor shape: {enhanced_tensor.shape}")
            noisy_tensor = noisy.reshape(1, -1)
            assert(enhanced_tensor.shape == noisy_tensor.shape)

            noisy_tensor = noisy_tensor[:, :file_len]
            enhanced_tensor = enhanced_tensor[:, :file_len]
            enhanced_tensor = normalize_estimates(enhanced_tensor, noisy_tensor)
            # clean_tensor = y[1].cpu()

            noisy_sisnr = -singlesrc_neg_sisdr(noisy_tensor, clean).mean()
            enhanced_sisnr = -singlesrc_neg_sisdr(enhanced_tensor, clean).mean()
            # print(f"unpadded sisnr: {enhanced_sisnr}, {noisy_sisnr}")
            self.noisy_sisnr_list.append(noisy_sisnr)
            self.sisnr_list.append(enhanced_sisnr)

            self.dnsmos_noisy += np.sum(self.dnsmos(noisy_tensor.cpu().data.numpy()), axis=0)
            self.dnsmos_clean += np.sum(self.dnsmos(clean.cpu().data.numpy()), axis=0)
            self.dnsmos_enhanced += np.sum(self.dnsmos(enhanced_tensor.cpu().data.numpy()), axis=0)

            self.noisy_pesqs += eval_pesq(noisy_tensor, clean, config.sample_rate, 'wb').cpu().data.numpy()
            self.clean_pesqs += eval_pesq(clean, clean, config.sample_rate, 'wb').cpu().data.numpy()
            self.enhanced_pesqs += eval_pesq(enhanced_tensor, clean, config.sample_rate, 'wb').cpu().data.numpy()

            self.noisy_stois += eval_stoi(noisy_tensor, clean, config.sample_rate).cpu().data.numpy()
            self.clean_stois += eval_stoi(clean, clean, config.sample_rate).cpu().data.numpy()
            self.enhanced_stois += eval_stoi(enhanced_tensor, clean, config.sample_rate).cpu().data.numpy()
            
            self.composite_noisy += eval_composite(sample_rate=config.sample_rate, 
                                                   ref_wav=clean.squeeze().cpu().data.numpy(),
                                                   deg_wav=noisy_tensor.squeeze().cpu().data.numpy())
            self.composite_clean += eval_composite(sample_rate=config.sample_rate, 
                                                   ref_wav=clean.squeeze().cpu().data.numpy(),
                                                   deg_wav=clean.squeeze().cpu().data.numpy())
            self.composite_enhanced += eval_composite(sample_rate=config.sample_rate, 
                                                      ref_wav=clean.squeeze().cpu().data.numpy(),
                                                      deg_wav=enhanced_tensor.squeeze().cpu().data.numpy())

            if len(self.noisy_sisnr_list) % 5 == 0:
                mean_event_rates = np.mean(self.test_event_rates, axis=0)
                syn_ops = []
                for rate, module_syn_ops in zip(mean_event_rates, self.event_syn_ops):
                    syn_ops.append(rate * module_syn_ops)
                effective_synops_rate = (sum(syn_ops) + self.fixed_syn_ops + 10 * self.neuron_ops) / self.dt
                synops_delay_product = effective_synops_rate * self.latency
                sep_synops_rate = (sum(syn_ops) + self.fixed_sep_syn_ops + 10 * self.neuron_ops) / self.dt
                sep_synops_delay_product = sep_synops_rate * self.latency
                spike_synops_rate = (sum(syn_ops) + 10 * self.neuron_ops) / self.dt
                spike_synops_delay_product = spike_synops_rate * self.latency

                rank_print(f"total flops: {self.total_flops}")
                for module_idx, mean_event_rate in enumerate(mean_event_rates):
                    print(f"Avg event fire rate for module {module_idx}: {mean_event_rate}")
                rank_print(f'Avg power proxy (Effective SynOPS)   : {effective_synops_rate:.3f} ops/s')
                rank_print(f'Avg PDP proxy (SynOPS-delay product) : {synops_delay_product: .3f} ops')
                rank_print(f'Avg power proxy (Effective SynOPS, excluding encoder and decoder)   : {sep_synops_rate:.3f} ops/s')
                rank_print(f'Avg PDP proxy (SynOPS-delay product, excluding encoder and decoder) : {sep_synops_delay_product: .3f} ops')
                rank_print(f'Avg spike power proxy (Effective SynOPS)   : {spike_synops_rate:.3f} ops/s')
                rank_print(f'Avg spike PDP proxy (SynOPS-delay product) : {spike_synops_delay_product: .3f} ops')

                rank_print(f"sisnr mean: {torch.mean(torch.tensor(self.sisnr_list))}, {torch.mean(torch.tensor(self.noisy_sisnr_list))}")
                
                noisy_pesqs = self.noisy_pesqs / len(self.noisy_sisnr_list)
                clean_pesqs = self.clean_pesqs / len(self.noisy_sisnr_list)
                enhanced_pesqs = self.enhanced_pesqs / len(self.noisy_sisnr_list)
                rank_print(f"\nAvg pesq [noisy, clean, enhanced]: {noisy_pesqs}, {clean_pesqs}, {enhanced_pesqs}")

                noisy_stois = self.noisy_stois / len(self.noisy_sisnr_list)
                clean_stois = self.clean_stois / len(self.noisy_sisnr_list)
                enhanced_stois = self.enhanced_stois / len(self.noisy_sisnr_list)
                rank_print(f"Avg stoi [noisy, clean, enhanced]: {noisy_stois}, {clean_stois}, {enhanced_stois}")

                composite_clean = self.composite_clean / len(self.noisy_sisnr_list)
                composite_noisy = self.composite_noisy / len(self.noisy_sisnr_list)
                composite_enhanced = self.composite_enhanced / len(self.noisy_sisnr_list)
                rank_print(f"\nAvg composite clean   [pesq, ovrl, sig, bak]: {composite_clean}")
                rank_print(f"Avg composite noisy   [pesq, ovrl, sig, bak]: {composite_noisy}")
                rank_print(f"Avg composite enhanced [pesq, ovrl, sig, bak]: {composite_enhanced}")

                dnsmos_clean = self.dnsmos_clean / len(self.sisnr_list)
                dnsmos_noisy = self.dnsmos_noisy / len(self.sisnr_list)
                dnsmos_enhanced = self.dnsmos_enhanced / len(self.sisnr_list)
                rank_print(f"\nAvg DNSMOS clean   [ovrl, sig, bak]: {dnsmos_clean}")
                rank_print(f"Avg DNSMOS noisy   [ovrl, sig, bak]: {dnsmos_noisy}")
                rank_print(f"Avg DNSMOS enhanced [ovrl, sig, bak]: {dnsmos_enhanced}")

                torchaudio.save(os.path.join(save_dir, f"fileid_{file_id}_enhanced.wav"), 
                                enhanced_tensor.cpu()[0].unsqueeze(0), 
                                config.sample_rate)
                torchaudio.save(os.path.join(save_dir, f"fileid_{file_id}_noisy.wav"), 
                                noisy_tensor.cpu()[0].unsqueeze(0), 
                                config.sample_rate)
                torchaudio.save(os.path.join(save_dir, f"fileid_{file_id}_clean.wav"), 
                                clean.cpu()[0].unsqueeze(0), 
                                config.sample_rate)


    test_callback = TestCallback()
    # early_stop_callback = EarlyStopping(**config.EarlyStopping)
    checkpoint_callback = ModelCheckpoint(monitor=config.checkpoint.monitor, 
                                          save_top_k=config.checkpoint.save_top_k,
                                          save_last=config.checkpoint.save_last,
                                          filename=config.checkpoint.filename)
    trainer = pl.Trainer(callbacks=[checkpoint_callback, test_callback], **config.trainer)
    assert((not args.load_ckpt_path) and (not args.test_ckpt_path), 
           f"Please make sure not to set load_ckpt_path and test_ckpt_path together!")

    spike_net = StreamSpikeNet(input_dim, context_dim,
                        sr=config.sample_rate, 
                        L=args.L, stride=args.stride,
                        N=args.N, B=args.B, H=args.H, X=args.X,
                        learning_rate=config.optim.lr)

    print(spike_net)
    # import torch._dynamo as dynamo
    # dynamo.config.cache_size_limit = 128
    # spike_net = torch.compile(spike_net)
    
    # spike_net = SDNN(learning_rate=config.optim.lr)

    if args.test_ckpt_path is None:  # training
        if args.load_ckpt_path:
            load_ckpt_path = os.path.join(script_dir, args.load_ckpt_path)
            rank_print(f"loading model from \"{load_ckpt_path}\"")
        else:
            load_ckpt_path = None
            rank_print(f"training from scratch")

        start = time.time()
        rank_print(f"training starts at: {time.asctime(time.localtime(start))}")
        trainer.fit(spike_net, train_dataloader, valid_dataloader, ckpt_path=load_ckpt_path)
        end = time.time()
        rank_print(f"training ends at: {time.asctime(time.localtime(end))}, time elapsed {(end-start)/60:.2f} min")
        
        rank_print(f"\n\ntesting with best:")
        trainer.test(spike_net, ckpt_path="best", dataloaders=test_dataloader)
        
        # rank_print(f"\n\ntesting with last:")
        # trainer.test(spike_net, ckpt_path="last", dataloaders=test_dataloader)
    else:  # testing
        test_ckpt_path = os.path.join(script_dir, args.test_ckpt_path)
        rank_print(f"test_ckpt_path: {test_ckpt_path}")
    
        trainer.test(spike_net, ckpt_path=test_ckpt_path, dataloaders=test_dataloader)


# guard in the main module to avoid creating subprocesses recursively. 
# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ == '__main__':
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    torch.set_float32_matmul_precision('high')
    start_func()