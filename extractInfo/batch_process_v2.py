import os
from processors_v2 import TranscriptProcessor

dir = "./Vault/testScripts"

extractor = TranscriptProcessor()
extractor.batch_process(dir, 10)
