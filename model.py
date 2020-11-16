import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from pytorch_transformers import BertPreTrainedModel, BertModel
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class DomainPredictionHead(nn.Module):
    def __init__(self, config):
        super(DomainPredictionHead, self).__init__()
        self.decoder = nn.Linear(config.hidden_size, 2)

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DepBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(DepBertPredictionHeadTransform, self).__init__()
        self.child_transform = nn.Linear(config.hidden_size, int(config.hidden_size/3))
        self.head_transform = nn.Linear(config.hidden_size, int(config.hidden_size/3))
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, child_hidden_states, head_hidden_states):
        child = self.transform_act_fn(self.child_transform(child_hidden_states))
        head = self.transform_act_fn(self.head_transform(head_hidden_states))
        hidden_states = torch.cat([child, head, child.mul(head)], dim=-1)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertDepPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertDepPredictionHead, self).__init__()
        self.transform = DepBertPredictionHeadTransform(config)
        self.decoder_dim = config.hidden_size
        relation_number = 47
        self.decoder = nn.Linear(self.decoder_dim,
                                 relation_number,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(relation_number))

    def forward(self, child_hidden_states, head_hidden_states):
        hidden_states = self.transform(child_hidden_states, head_hidden_states)  # (batch, seq_len, hidden_dim)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPosPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertPosPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)  # 一个等维度的非线性变换

        self.decoder_dim = config.hidden_size
        self.decoder = nn.Linear(config.hidden_size,
                                 83,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(83))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class DepBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(DepBertPreTrainingHeads, self).__init__()
        self.dep_predictions = BertDepPredictionHead(config)
        self.pos_predictions = BertPosPredictionHead(config)

    def forward(self, sequence_output, child_output, head_output):
        dep_relationship_scores = self.dep_predictions(child_output, head_output)
        tag_prediction_scores = self.pos_predictions(sequence_output)
        return tag_prediction_scores, dep_relationship_scores


class InstanceLoss(nn.Module):
    def __init__(self, eps=0.2, reduction='mean', ignore_index=-1):
        super(InstanceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, target, weights):
        '''
        input: (batch*seq_len, C) 
        target: (batch*seq_len) labels for every token
        weights: (batch*seq_len) weights for every token
        '''
        log_preds = F.log_softmax(inputs, dim=-1)
        loss = F.nll_loss(log_preds, target, reduction='none', ignore_index=self.ignore_index)

        mask = target.squeeze().float()
        mask = mask.masked_fill(mask==0.0, self.eps)
        mask = mask.masked_fill(mask != self.eps, 1.0)
        # Only relying on domain probability will select most of the words with O tags (more than 80%). 
        # For example, most stop words have the same distribution across domains, but they are meaning less.
        # We use the mask vector to rescale the probability distribution of words between domains.
        weights = mask + weights
        weights = F.softmax(weights, dim=-1)
        return torch.matmul(weights.view(-1), loss)

        
class ABSABert(BertPreTrainedModel):
    def __init__(self, config):
        super(ABSABert, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.cls = DepBertPreTrainingHeads(config)
        self.domain_cls = DomainPredictionHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_tags=None, head_tokens_index=None, dep_relation_label=None,
                masked_tag_labels=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                domain_label=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]  # sequence_output: (batch, seq_len, hidden)
        sequence_output = self.dropout(sequence_output)

        if masked_tag_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            now_batch = sequence_output.size()[0]
            head_output = torch.stack([sequence_output[i, head_tokens_index[i], :] for i in range(now_batch)],
                                      dim=0)
            tag_prediction_scores, dep_relationship_scores = self.cls(sequence_output, sequence_output, head_output)  # 拼接之后，送入预测层

            masked_tag_loss = loss_fct(tag_prediction_scores.view(-1, 83), masked_tag_labels.view(-1))
            dep_relationship_loss = loss_fct(dep_relationship_scores.view(-1, 47), dep_relation_label.view(-1))

            total_loss = masked_tag_loss + 0.1 * dep_relationship_loss
            return total_loss
        elif domain_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            max_len = sequence_output.size()[1]
            avg_pool = nn.functional.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=max_len).squeeze()
            domain_predicte_scores = self.domain_cls(avg_pool)
            if len(domain_predicte_scores.size()) == 1:
                domain_predicte_scores = domain_predicte_scores.unsqueeze(0)
            domain_loss = loss_fct(domain_predicte_scores, domain_label)
            return domain_loss
        else:
            logits = self.classifier(sequence_output)
            loss_fct = InstanceLoss()
            if labels is not None:
                with torch.no_grad():
                    factors = self.domain_cls(sequence_output)
                    factors = F.softmax(factors, dim=-1)
                    factors, _ = torch.min(factors, dim=-1, keepdim=True)
                    factors = factors.squeeze()  # batch*seq_len
                    factors = F.softmax(factors, dim=-1)
                    
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), factors.view(-1))
                return loss, factors
            else:
                return logits
