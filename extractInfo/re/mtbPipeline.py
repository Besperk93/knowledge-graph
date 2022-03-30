from transformers import pipeline

class mtb_pipeline(pipeline):

    def _sanitize_parameters(self, **kwargs):
        pass

    def preprocess(self, inputs, maybe_arg=2):
        # Handle the inputs from the transcripts
        pass

    def _forward(self, model_inputs):
        # all connections to the underlying model should be here
        # calling "forward" will trigger this function with safeguards
        pass

    def postprocess(self, model_outputs):
        pass
