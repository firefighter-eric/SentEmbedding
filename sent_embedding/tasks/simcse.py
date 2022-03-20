import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from scipy.stats import spearmanr
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from ..models.sent_models import SentVector, Similarity


class SimCSE(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = SentVector(self.args)
        if self.args.mlm_flag:
            self.mlm = BertLMPredictionHead(self.args)
        self.sim = Similarity(self.args.temp)
        self.loss = CrossEntropyLoss()

    def forward(self, s1, s2, mlm_label=None):
        outputs_1, pooler_output_1 = self.model(s1)
        outputs_2, pooler_output_2 = self.model(s2)
        batch_size = outputs_1.size(0)

        # sim loss
        z1 = self.all_gather(pooler_output_1)
        z2 = self.all_gather(pooler_output_2)
        sim = self.sim(z1, z2)
        sim_label = torch.ones(batch_size, dtype=torch.long, device=self.device)
        sim_loss = self.sim(sim, sim_label)

        # mlm loss
        if self.args.mlm_flag:
            m1 = self.mlm(outputs_1)
            m2 = self.mlm(outputs_2)
            mlm_loss = self.loss(m1, mlm_label) + self.loss(m2, mlm_label)
        else:
            mlm_loss = 0

        loss = sim_loss + self.args.mlm_weight * mlm_loss
        return sim, sim_loss, mlm_loss, loss

    def training_step(self, batch, batch_idx):
        sim, sim_loss, mlm_loss, loss = self(batch)
        self.log('train/sim_loss', sim_loss)
        self.log('train/mlm_loss', mlm_loss)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])

    def validation_step(self, batch, batch_idx):
        sim, sim_loss, mlm_loss, loss = self(batch)
        self.log('val/sim_loss', sim_loss)
        self.log('val/mlm_loss', mlm_loss)
        self.log('val/loss', loss)

        batch_size = sim.size(0)
        similarity_flatten = sim.view(-1).tolist()
        label_flatten = torch.eye(batch_size).view(-1).tolist()
        spearman_corr = spearmanr(similarity_flatten, label_flatten).correlation()
        self.log('val/spearman_corr', spearman_corr)
        return loss

    def configure_optimizers(self):
        lr = self.args.lr
        weight_decay = self.args.weight_decay
        warm_up_epochs = self.args.warm_up_epochs

        def lr_foo(epoch):
            if epoch < self.args.warm_up_epochs:
                lr_scale = 0.1 ** (warm_up_epochs - epoch)  # warm up lr
            else:
                lr_scale = 0.95 ** epoch
            return lr_scale

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_foo)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
