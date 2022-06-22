import torch
import numpy as np
import torch.nn as nn
import torchio as tio
from medpy.metric.binary import dc

from utils.utils import AverageMeter, determine_device


def D(p, z):
    return - torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()


def train(model, train_generator, optimizer, criterion, logger, config, epoch):
    model.train()
    segnet = model['segnet']
    projections = model['projections']
    predictions = model['predictions']
    losses = AverageMeter()
    # if scaler is not supported, it switches to default mode, the training can also continue
    scaler = torch.cuda.amp.GradScaler()
    num_iter = config.TRAIN.NUM_BATCHES
    for idx in range(num_iter):
        data_dict = next(train_generator)
        data = data_dict['data']
        label = data_dict['label']
        if config.TRAIN.PARALLEL:
            devices = config.TRAIN.DEVICES
            data = data.cuda(devices[0])
            label = [l.cuda(devices[0]) for l in label]
        else:
            device = determine_device()
            data = data.to(device)
            label = [l.to(device) for l in label]
        # run training
        with torch.cuda.amp.autocast():
            x, rec = data[:, :-1], data[:, -1:]
            out, embeds = segnet(x, rec)
            l_dc = criterion(out, label[0])
            # simsiam
            l_sim = .0
            for i in range(4):
                mask = 1 - label[i]
                v1, v2 = projections[i](embeds[i][0]*mask), projections[i](embeds[i][1]*mask)
                p1, p2 = predictions[i](v1), predictions[i](v2)
                l_sim += .5 * D(p1, v2) + .5 * D(p2, v1)
            loss = l_dc + l_sim
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()

        if idx % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, i, num_iter,
                    loss = losses,
                )
            logger.info(msg)


def inference(model, dataset, logger, config):
    model.eval()
    
    perfs = [AverageMeter() for _ in range(2)]
    nonline = nn.Softmax(dim=1)
    scores = {}
    for case in dataset:
        patch_size = config.INFERENCE.PATCH_SIZE
        patch_overlap = config.INFERENCE.PATCH_OVERLAP
        # torchio does not support 2d slice natively, it can only treat it as pseudo 3d patch
        patch_size = [1] + patch_size
        patch_overlap = [0] + patch_overlap
        # data shape cannot be smaller than patch size, maybe pad is needed
        target_shape = np.max([patch_size, case['data'].shape[1:]], 0)
        transform = tio.CropOrPad(target_shape)
        case = transform(case)
        # sliding window sampler
        sampler = tio.inference.GridSampler(case, patch_size, patch_overlap)
        loader = torch.utils.data.DataLoader(sampler, config.INFERENCE.BATCH_SIZE)
        aggregator = tio.inference.GridAggregator(sampler, 'average')

        with torch.no_grad():
            for data_dict in loader:
                data = data_dict['data'][tio.DATA]
                label = data_dict['label'][tio.DATA]
                data = data.squeeze(2)
                label = label.squeeze(2)
                if config.TRAIN.PARALLEL:
                    devices = config.TRAIN.DEVICES
                    data = data.cuda(devices[0])
                    label = label.cuda(devices[0])
                else:
                    device = determine_device()
                    data = data.to(device)
                    label = label.to(device)
                with torch.cuda.amp.autocast():
                    x, rec = data[:, :-1], data[:, -1:]
                    out = model(x, rec)
                    out = nonline(out)
                locations = data_dict[tio.LOCATION]
                # I love and hate torchio ...
                out = out.unsqueeze(2)
                aggregator.add_batch(out, locations)
            # form final prediction
            pred = aggregator.get_output_tensor()
            pred = torch.argmax(pred, dim=0).cpu().numpy()
            label = case['label'][tio.DATA][0].numpy() > 0  # only wt segmentation is supported
            name = case['name']
            # quantitative analysis
            # only dice score is computed by default, you can also add hd95, assd and sensitivity et al
            scores[name] = {}
            for c in np.unique(label):
                scores[name][int(c)] = dc(pred==c, label==c)
                perfs[int(c)].update(scores[name][c])
        del case
    logger.info('------------ dice scores ------------')
    logger.info(scores)
    for c in range(2):
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')
    perf = np.mean([perfs[c].avg for c in range(1, 2)])
    return perf
