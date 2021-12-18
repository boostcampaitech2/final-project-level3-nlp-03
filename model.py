import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
from typing import List

class CustomTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size // 2, bidirectional=True, batch_first=True)
        self.dropout=nn.Dropout(classifier_dropout)
        self.classifier=nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = outputs[0]
        hidden, (last_hidden, last_cell) = self.lstm(output)
        sequence_output = self.dropout(hidden)        
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CustomTokenClassificationCRF(CustomTokenClassification):
    def __init__(self, model_name, num_labels):
        super().__init__(model_name, num_labels)

        self.crf = CRF(num_tags=self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = outputs[0]

        hidden, (last_hidden, last_cell) = self.lstm(output)
        sequence_output = self.dropout(hidden)        

        logits = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood, logits = self.crf(logits, labels), self.crf.decode(logits)
            loss = -1 * log_likelihood
        else:
            logits = self.crf.decode(logits)

        logits = reverse_argmax(logits, self.num_labels)        
        
        return TokenClassifierOutput(
            loss=loss,
            logits=torch.tensor(logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def reverse_argmax(logits:List[List[int]], num_labels:int)-> List[List[List[int]]]:
    # logits needs to be [32, 256, -1] but crf logits returns [32,256]
    # pytorch 
    reversed_list = []
    for logit in logits:
        temp_logit = []
        for label in logit:
            temp_list = [0]*num_labels
            temp_list[label] = 1
            temp_logit.append(temp_list)
        reversed_list.append(temp_logit)

    return reversed_list