import torch
import torchio as tio
import torch.nn.functional as F

from utils.utils import AverageMeter
from models.network import reparameterize


def kl(mu, log_var):
    loss = - 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
    return loss


def train(model, trainloader, optimE, optimD, logger, config, epoch):
    encoder, decoder = model

    ls_enc = AverageMeter()
    ls_dec = AverageMeter()
    alpha, beta, eta = 0.5, 0.04, 120
    for idx, data in enumerate(trainloader):
        real = data['data'][tio.DATA].to(torch.device(config.DEVICE)).squeeze(2)
        batch_size = real.size(0)
        optimE.zero_grad()
        optimD.zero_grad()
        # real : real image
        # rec  : reconstructed real image
        # pseudo : sampled pseudo image  
        
        # encoder training
        encoder.requires_grad = True
        decoder.required_grad = False
        real_mu, real_logvar = encoder(real)
        real_code = reparameterize(real_mu, real_logvar)
        rec = decoder(real_code)
        # send rec to encoder again
        rec_mu, rec_logvar = encoder(rec.detach())
        # randomly sample a latent code
        pseudo_code = torch.randn_like(real_code)
        pseudo = decoder(pseudo_code)
        pseudo_mu, pseudo_logvar = encoder(pseudo.detach())
        # encoder loss: regularization loss / adversarial loss / reconstruction loss
        l_enc = kl(real_mu, real_logvar) + \
            alpha * (F.relu(eta - kl(rec_mu, rec_logvar)) + F.relu(eta - kl(pseudo_mu, pseudo_logvar))) +\
                beta * F.mse_loss(rec, real)
        l_enc.backward(retain_graph=True)
        optimE.step()
        ls_enc.update(l_enc, batch_size)

        # decoder training
        encoder.requires_grad = False
        decoder.required_grad = True
        rec = decoder(real_code.detach())
        pseudo = decoder(pseudo_code)
        rec_mu, rec_logvar = encoder(rec)
        pseudo_mu, pseudo_logvar = encoder(pseudo)
        # decoder loss: adversarial loss / reconstruction
        l_dec = alpha * (kl(rec_mu, rec_logvar) + kl(pseudo_mu, pseudo_logvar)) + beta * F.mse_loss(rec, real)
        l_dec.backward()
        optimD.step()
        ls_dec.update(l_dec, batch_size)
        
        del data
        if idx % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Enc {loss1.val:.3f} ({loss1.avg:.3f})' \
                'Dec {loss2.val:.3f} ({loss2.avg:.3f})'.format(
                    epoch, idx, len(trainloader),
                    loss1 = ls_enc,
                    loss2 = ls_dec
                )
            logger.info(msg)
