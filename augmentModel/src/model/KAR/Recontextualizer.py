import torch
import torch.nn as nn

from transformers import BertConfig
from transformers.models.bert.modeling_bert import (BertEncoder, BertSelfOutput,
            BertIntermediate, BertOutput
            )

from .Utilities import set_start_weights


class WordAttention(nn.Module):

    # Establish a key, query and value as a linear function of the entity embeddings and hidden states

    def __init__(self, config):
        super(WordAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, entity_embeddings, entity_mask):
        # NOTE: Most of this is OMH, clone if short on time
        # apply linear transformation
        mixed_key_layer = self.key(entity_embeddings)
        mixed_query_layer = self.query(hidden_states)
        mixed_value_layer = self.value(entity_embeddings)
        # transpose for further computations
        key_layer = self.transpose_for_scores(mixed_key_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # compute raw attention scores
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # apply attention mask
        attention_mask = extend_mask(entity_mask)
        attention_scores = attention_scores + attention_mask
        # apply softmax and dropout
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # contextualize values
        context_layer = attention_probs @ value_layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], self.all_head_size)
        # return contextualized values and attention probs
        return context_layer, attention_probs



class SpanAttention(nn.Module):

    # Pass the outputs from the WordAttention layer through a Bert Output Layer

    def __init__(self, config):
        super(SpanAttention, self).__init__()
        # create modules
        self.attention = WordAttention(config)
        self.output = BertSelfOutput(config)

        # initialize weights
        set_start_weights(self.attention, 0.02, (WordAttention, ))
        set_start_weights(self.output, 0.02)


    def forward(self, hidden_states, entity_embeddings, entity_mask):
        span_output, attention_probs = self.attention(hidden_states, entity_embeddings, entity_mask)
        attention_output = self.output(span_output, hidden_states)
        return attention_output, attention_probs




class AttentionLayer(nn.Module):

    # Pass hidden_states, entity embeddings and an entity mask through span and word attention layers

    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        # create modules
        # NOTE: We may need to start thinking about preparing KAR output for GPT (rather than BERT) here
        self.attention = SpanAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # initialize weights
        set_start_weights(self.intermediate, 0.02)
        set_start_weights(self.output, 0.02)


    def forward(self, hidden_states, entity_embeddings, entity_mask):
        attention_output, attention_probs = self.attention(hidden_states, entity_embeddings, entity_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class Recontextualizer(nn.Module):

    def __init__(self, kb, span_attention_config:BertConfig):
        super(Recontextualizer, self).__init__()
        # create modules
        span_attention_config.hidden_size = kb.embedding_dimensions
        self.span_attention_layer = AttentionLayer(span_attention_config)

    def forward(self, enhanced_span_representations, h_projections, mention_mask):
        span_attention_output, _ = self.span_attention_layer(h_projections, enhanced_span_representations, mention_mask)
        return span_attention_output
