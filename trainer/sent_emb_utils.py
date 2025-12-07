import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.nn import functional as F

logger = logging.getLogger('fpd_sent_emb_main')

class SimilarityModel(nn.Module):
    def __init__(self, H, D):
        """
        H: Input feature dimension
        D: Transformed feature dimension (embedding space)
        """
        super().__init__()
        
        # Learnable transformation networks for A and B
        self.f_A = nn.Sequential(
            nn.Linear(H, D),
            nn.ReLU(),
            nn.Linear(D, D)  # Output embedding of size D
        )
        
        self.f_B = nn.Sequential(
            nn.Linear(H, D),
            nn.ReLU(),
            nn.Linear(D, D)  # Output embedding of size D
        )
        
    def forward(self, query, key):
        # Transform queries and keys
        query_emb = self.f_A(query)  # [M, D]
        key_emb = self.f_B(key)      # [N, D]

        # Compute similarity as the inner product (dot product)
        similarity = query_emb @ key_emb.T  # [M, N]

        return similarity
    
    def transform_query(self, query):
        query_emb = self.f_A(query)  
        return query_emb
    
    def transform_key(self, key):
        key_emb = self.f_B(key)
        return key_emb

def compute_rmse_tensor(gts, preds):
    rmse = torch.sqrt(F.mse_loss(preds, gts)).item()
    return rmse

def bag_ocl_reps(ocl_reps_full, ratio):
    rets = []
    for k, reps in ocl_reps_full.items():
        n = min(1, int(reps.shape[0] * ratio))
        indices = torch.randperm(reps.shape[0])[:n]
        rets.append(reps[indices].mean(0))
    rets = torch.stack(rets)
    return rets

def train_similarity_model(config, ocl_reps_full, pt_reps, gt, gt_bias, test_reps=None, test_gts=None, test_bias=None):
    H, D = config.fpd.sent_input_dim, config.fpd.sent_hidden_dim
    model = SimilarityModel(H, D)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()  # Assuming R contains real similarity values

    regression_target = gt - gt_bias if gt_bias is not None else gt
    
    ocl_reps_full = {k: v.cuda() for k, v in ocl_reps_full.items()}
    pt_reps, regression_target = pt_reps.cuda(), regression_target.cuda()

    num_epochs = 5000
    val_every = 100

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # bag according to the ratio
        ocl_reps = bag_ocl_reps(ocl_reps_full, config.fpd.bagging_ratio)

        pred_sim = model(ocl_reps, pt_reps)  # Predict similarity matrix S (shape [M, N])
        loss = loss_fn(pred_sim, regression_target)  # Compare with ground-truth similarity matrix R
        
        loss.backward()
        optimizer.step()

        rmse = loss.item() ** 0.5

        if epoch and epoch % val_every == 0 and test_gts is not None:
            with torch.no_grad():
                test_output = infer_sim_model(config, model, test_reps, pt_reps, gt_bias=test_bias)
                test_rmse = compute_rmse_tensor(test_gts, test_output)
                logger.info(f'epoch {epoch} test rmse {test_rmse}')
            logger.info(f'epoch {epoch} train rmse {rmse}')
    return model

def infer_sim_model(config, model, ocl_reps, pt_reps, gt_bias=None):
    model = model.cuda()
    ocl_reps, pt_reps = ocl_reps.cuda(), pt_reps.cuda()
    rep_sims = model(ocl_reps, pt_reps)
    rep_sims = rep_sims.cpu()
    
    if gt_bias is not None:
        ret = rep_sims + gt_bias
    else:
        ret = rep_sims

    return ret

def load_similarity_model(config):
    H, D = config.fpd.sent_input_dim, config.fpd.sent_hidden_dim
    model = SimilarityModel(H, D)
    with open(config.fpd.sim_model_path,'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    return model

def save_similarity_model(config, model):
    state_dict = {k:v.cpu() for k,v in model.state_dict.items()}
    with open(config.fpd.sim_model_path,'wb') as wf:
        torch.save(state_dict, wf)