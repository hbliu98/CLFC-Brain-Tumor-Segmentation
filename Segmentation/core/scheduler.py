from torch.optim.lr_scheduler import LambdaLR


class PolyScheduler(LambdaLR):
    def __init__(self, optimizer, t_total, exponent=0.9, last_epoch=-1):
        self.t_total = t_total
        self.exponent = exponent
        super(PolyScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return (1 - step / self.t_total)**self.exponent
