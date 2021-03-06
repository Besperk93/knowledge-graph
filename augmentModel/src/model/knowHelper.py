import torch
import torch.nn as nn


class KnowGPT2Helper(object):

    def __init__(self, kbs, config):
        self.kbs = nn.ModuleList([None for _ in range(self.config.num_hidden_layers)])
        self.pretrained_path = config.pretrained_model_name_or_path
        for layer, kb_config in config.kbs.items():
            layer = int(layer)
            # read from config
            kb_type, kar_kwargs = kb_config['type'], kb_config['kar_kwargs']
            # create path to save directory for the knowledge base
            kb_pretrained_path = os.path.join(pretrained_path, '%i-%s' % (layer, kb_type))
            # load knowledge base
            knowledge_base_type = KnowledgeBaseRegistry.instance.get_type_from_name(kb_type)
            kb = knowledge_base_type.load(kb_pretrained_path, kb_config)
            # add knowledge base to encoder
            self._add_knowledge(layer, kb, **kar_kwargs)

    def _add_knowledge(self, layer:int, kb:KnowledgeBase, **kar_kwargs) -> None:
        # check if kb is of correct type
        if not isinstance(kb, KnowledgeBase):
            raise RuntimeError("%s must inherit KnowledgeBase" % kb.__class__.__name__)
        # check if layer already has a kb
        if self.kbs[layer] is not None:
            raise RuntimeError("There already is a knowledge base at layer %i" % layer)
        # add knowledge base to layer
        self.kbs[layer] = KAR(kb, self.config, **kar_kwargs)


    def add_knowledge(self, layer:int, kb:KnowledgeBase, max_mentions=15, max_mention_span=5, max_candidates=10, threshold=None) -> KAR:
        """ add a knowledge bases in between layer and layer+1 """

        # span-encoder-config
        span_encoder_config = {
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 1024
        }
        # span-attention-config
        span_attention_config = {
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 1024
        }

        # build keyword arguments for the kar module
        kar_kwargs = {
            'span_encoder_config': span_encoder_config,
            'span_attention_config': span_attention_config,
            'max_mentions': max_mentions,
            'max_mention_span': max_mention_span,
            'max_candidates': max_candidates,
            'threshold':threshold
        }
        # add knowledge base
        self._add_knowledge(layer, kb, **kar_kwargs)
        self.config.add_kb(layer, kb, kar_kwargs)
        # return knowledge base
        return self.kbs[layer]

    def save_kbs(self, save_directory):
        # save all knowledge bases
        for i, kar in enumerate(self.kbs):
            if kar is not None:
                # create a subsidrectory for the knowledge base
                kb_name = KnowledgeBaseRegistry.instance.get_name_from_type(kar.kb.__class__)
                kb_save_directory = os.path.join(save_directory, "%i-%s" % (i, kb_name))
                # save knowledge base to sub-directory
                os.makedirs(kb_save_directory, exist_ok=True)
                kar.kb.save(kb_save_directory)

    def freeze_layers(self, layer:int):
        """ Freeze all layers after given index """
        for n, p in self.named_parameters():
            # find layer index of parameter
            m = re.search(r".\d+.", n)
            l = int(m.group().replace('.', ''))
            # freeze if layer is before the given one
            if l <= layer:
                # exclude knowledge bases
                if n.startswith('kbs') and (l == layer):
                    # unfreeze
                    p.requires_grad_(True)
                else:
                    # freeze parameter
                    p.requires_grad_(False)
            else:
                # unfreeze
                p.requires_grad_(True)

    def get_kb_caches(self):
        """ get current caches of all knowledge bases """
        return [kb.cache if kb is not None else None for kb in self.kbs]

    def clear_kb_caches(self):
        """ reset all caches of all knowledge bases """
        for kb in self.kbs:
            if kb is not None:
                kb.clear_cache()

    def prepare_kbs(self, batch_tokens:list):
        """ prepare all knowledge bases for next forward pass.
            Basically computes and sets all caches for the given batch.

            Returns a list of mention-candidates dicts for each layer and each token-sequence.
            return = ([dict, ..., dict], ..., [dict, ..., dict])
        """
        # reset and set caches
        self.clear_kb_caches()
        caches, candidates = zip(*[self.build_kb_caches(tokens, True) for tokens in batch_tokens])
        self.stack_kb_caches(*caches)
        # return all candidates
        return candidates

    def build_kb_caches(self, tokens:list, output_candidates:bool=True):
        """ Get cache for each knowledge base from tokens.

            Return a list of caches, one for each knowledge base and None for layers without one.
            If output_candidates is set it also returns a mention-candidates dict for each layer.
            return = [cache, ..., cache], [dict, ..., dict]
        """
        caches, candidates = zip(*[kb.get_cache_and_mention_candidates(tokens) if kb is not None else (None, None) for kb in self.kbs])
        if output_candidates:
            return caches, candidates
        return caches

    def stack_kb_caches(self, *all_caches):
        """ Stack multiple caches for each knowledge base.
            all_caches must be a list of caches where each list contains
            one cache per knowledge base and None for layers without one

            all_caches = *([cache, ..., cache], ..., [cache, ..., cache])
        """
        # loop over all caches per knowledge base
        for kb, caches in zip(self.kbs, zip(*all_caches)):
            if kb is not None:
                assert all([t is not None for t in caches])
                kb.stack_caches(*caches)
    def set_valid_kb_caches(self, *caches):
        """ Set the caches of each valid knowledge base.
            The Function expects only the caches of valid kbs in the correct order as input.
        """

        # clear all caches and get valid knowledge bases
        self.clear_kb_caches()
        kbs = [kb for kb in self._knowbert_encoder.kbs if kb is not None]
        # must provide a cache for each knowledge base
        assert len(kbs) == len(caches)
        # set all caches
        for kb, cache in zip(kbs, caches):
            kb.stack_caches(cache)

    def prepare_kbs(self, tokens:list):
        """ prepare all knowledge bases for next forward pass """
        return self._knowbert_encoder.prepare_kbs(tokens)

    def save_pretrained(self, save_directory:str):
        # save model and knowledge bases
        super().save_pretrained(save_directory)
        self.save_kbs(save_directory)


    # *** forward ***

    def forward(self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        output_linking_scores=True,
        return_dict=False,
    ):

        # prepare outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_linking_scores = () if output_linking_scores else None
        # running loss of knowledge bases
        running_kb_loss = 0

        # pass through each layer
        for i, layer_module in enumerate(self.layer):
            # save all hidden states if asked for
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )

            else:
                # pass through layer
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            # get hidden state
            hidden_states = layer_outputs[0]

            # save attention values if asked for
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # apply knowledge base for layer if there is one
            if self.kbs[i] is not None:
                hidden_states, linking_scores, kb_loss = self.kbs[i].forward(hidden_states)
                running_kb_loss += kb_loss
                # add linking scores to tuple
                if output_linking_scores:
                    all_linking_scores = all_linking_scores + (linking_scores,)

            # add None to linking scores tuple
            elif output_linking_scores:
                all_linking_scores = all_linking_scores + (None,)


        # add very last hidden state to tuple
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # return tuple
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions, all_linking_scores] if v is not None) + (running_kb_loss,)
        # return dict/output-class
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )
