import os
from processors import TranscriptProcessor

dir = "Vault/transcripts"

extractor = TranscriptProcessor()
extractor.batch_process(dir, 50)
