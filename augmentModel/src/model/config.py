from transformers import GPT2Config
from .knowledgeBase import KnowledgeBase

class KnowGPT2Config(GPT2Config):

    def __init__(       self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        kbs={},
        gradient_checkpointing=False,
        pretrained_model_name_or_path=None,
        **kwargs):

        # initialize bert config from super class:
        GPT2Config.__init__(self,
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            scale_attn_by_inverse_layer_idx=scale_attn_weights,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
            **kwargs,
        )
        # save knowledge configurations
        self.kbs = kbs
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def add_kb(self, layer:int, kb:KnowledgeBase, kar_kwargs:dict) -> None:
        # check if kb is of correct type
        if not isinstance(kb, KnowledgeBase):
            raise RuntimeError("%s must inherit KnowledgeBase" % kb.__class__.__name__)
        # check if layer already has a kb
        if self.kbs.get(layer, None) is not None:
            raise RuntimeError("There already is a knowledge base at layer %i" % layer)

        # build full config
        config = kb.config
        config.update({'kar_kwargs': kar_kwargs})
        # add knowledge base to config
        # Note: this seems to be adding the kb.config to the kb in the dictionary self.kbs{}
        self.kbs[layer] = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path:str, **kwargs):
        # check class
        if not issubclass(cls, KnowGPT2Config):
            raise RuntimeError("Knowbert Configuration must inhert KnowBertConfig! (Got %s)" % cls.__name__)
        # create configuration from model name or path
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        kwargs.update({'pretrained_model_name_or_path': pretrained_model_name_or_path})
        return cls.from_dict(config_dict, **kwargs)
