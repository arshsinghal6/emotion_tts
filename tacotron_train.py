import os
import argparse

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import data_functions
import models
import loss_functions   
from train import parse_args
from tacotron2.data_function import TextMelLoader

def adjust_learning_rate(iteration, epoch, optimizer, learning_rate,
                         anneal_steps, anneal_factor, rank):

    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p+1

    if anneal_factor == 0.3:
        lr = learning_rate*((0.1 ** (p//2))*(1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate*(anneal_factor ** p)

    if optimizer.param_groups[0]['lr'] != lr:
        print(step=(epoch, iteration), data={'learning_rate changed': str(optimizer.param_groups[0]['lr'])+" -> "+str(lr)})

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(model, optimizer, scaler, epoch, config, output_dir,
                    model_name, local_rank, world_size):

    random_rng_state = torch.random.get_rng_state().cuda()
    cuda_rng_state = torch.cuda.get_rng_state(local_rank).cuda()

    random_rng_states_all = [torch.empty_like(random_rng_state) for _ in range(world_size)]
    cuda_rng_states_all = [torch.empty_like(cuda_rng_state) for _ in range(world_size)]

    # if world_size > 1:
    #     dist.all_gather(random_rng_states_all, random_rng_state)
    #     dist.all_gather(cuda_rng_states_all, cuda_rng_state)
    # else:
    random_rng_states_all = [random_rng_state]
    cuda_rng_states_all = [cuda_rng_state]

    random_rng_states_all = torch.stack(random_rng_states_all).cpu()
    cuda_rng_states_all = torch.stack(cuda_rng_states_all).cpu()

    if local_rank == 0:
        checkpoint = {'epoch': epoch,
                      'cuda_rng_state_all': cuda_rng_states_all,
                      'random_rng_states_all': random_rng_states_all,
                      'config': config,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scaler': scaler.state_dict()}

        checkpoint_filename = "checkpoint_{}_{}.pt".format(model_name, epoch)
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        print("Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path))
        torch.save(checkpoint, checkpoint_path)

        symlink_src = checkpoint_filename
        symlink_dst = os.path.join(
            output_dir, "checkpoint_{}_last.pt".format(model_name))
        if os.path.exists(symlink_dst) and os.path.islink(symlink_dst):
            print("Updating symlink", symlink_dst, "to point to", symlink_src)
            os.remove(symlink_dst)

        os.symlink(symlink_src, symlink_dst)

def validate(model, criterion, valset, epoch, batch_iter, batch_size,
             world_size, collate_fn, distributed_run, perf_bench, batch_to_gpu, amp_run):
    """Handles all the validation scoring and printing"""
    
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                sampler=None,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn,
                                drop_last=(True if perf_bench else False))

        val_loss = 0.0
        num_iters = 0
        val_items_per_sec = 0.0
        for i, batch in enumerate(val_loader):

            x, y, num_items = batch_to_gpu(batch)
            #AMP upstream autocast
            with torch.cuda.amp.autocast(enabled=amp_run):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            reduced_val_loss = loss.item()
            reduced_num_items = num_items.item()
            val_loss += reduced_val_loss
            
            num_iters += 1

        val_loss = val_loss/num_iters
        val_items_per_sec = val_items_per_sec/num_iters

        return val_loss, val_items_per_sec
    
def train(output_dir: str):
    writer = SummaryWriter(output_dir)
    
    # valset = data_functions.get_data_loader(
    #         MODEL_NAME, args.dataset_path, args.validation_files, args)
    batch_to_gpu = data_functions.get_batch_to_gpu(MODEL_NAME)

    local_rank, world_size = args.rank, args.world_size
    distributed_run = world_size > 1

    epoch = [0]
    start_epoch = epoch[0]
    iteration = 0
    
    epoch_loss = 0
    log_len = len(train_loader) // 30

    for epoch in tqdm(range(start_epoch, args.epochs)):
        # used to calculate avg items/sec over epoch
        reduced_num_items_epoch = 0
        
        num_iters = 0
        reduced_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader)):
            adjust_learning_rate(iteration, epoch, optimizer, args.learning_rate,
                                        args.anneal_steps, args.anneal_factor, local_rank)

            model.zero_grad()
            x, y, num_items = batch_to_gpu(batch)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            reduced_loss = loss.item()
            reduced_num_items = num_items.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")
        
            epoch_loss += loss
            num_iters += 1

            # accumulate number of items processed in this epoch
            reduced_num_items_epoch += reduced_num_items

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_thresh)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_thresh)
                optimizer.step()

            model.zero_grad(set_to_none=True)

            if i % log_len == 0:
                writer.add_scalar("Step Loss Train", loss, i)

            val_loss, val_items_per_sec = validate(model, criterion, val_dataset, epoch,
                                                iteration, args.batch_size,
                                                world_size, collate_fn,
                                                distributed_run, args.bench_class=="perf-train",
                                                batch_to_gpu,
                                                args.amp)
            writer.add_scalar("Validation Loss Train", val_loss, epoch+1)


            # if (epoch % args.epochs_per_checkpoint == 0) and (args.bench_class == "" or args.bench_class == "train"):
            #     save_checkpoint(model, optimizer, scaler, epoch, model_config,
            #                     args.output, args.model_name, local_rank, world_size)

        writer.add_scalar("Epoch Loss Train", epoch_loss, epoch+1)
        epoch_loss = 0

MODEL_NAME = 'Tacotron2'
OUTPUT_DIR = './models/'
CPU_RUN = False

N_EMOTIONS = 5
SAMPLING_RATE = 16000

EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 64

args = argparse.ArgumentParser()
parser = parse_args(args)
parser = models.model_parser(MODEL_NAME, parser)
args = parser.parse_args(f'--epochs {N_EMOTIONS} -lr {LR} -bs {BATCH_SIZE} -m {MODEL_NAME} -o {OUTPUT_DIR} --sampling-rate {SAMPLING_RATE} --n-emotions {N_EMOTIONS}'.split())

train_dataset = TextMelLoader(dataset_path='', audiopaths_and_text='./data/train_mel_text_pairs.txt', args=args)
val_dataset = TextMelLoader(dataset_path='', audiopaths_and_text='./data/test_mel_text_pairs.txt', args=args)

collate_fn = data_functions.get_collate_function(
        MODEL_NAME, args.n_frames_per_step)
train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=args.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

model_config = models.get_model_config(MODEL_NAME, args)
model = models.get_model(MODEL_NAME, model_config, CPU_RUN,
                         uniform_initialize_bn_weight=not args.disable_uniform_initialize_bn_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

criterion = loss_functions.get_loss_function(MODEL_NAME, sigma=None)

train()