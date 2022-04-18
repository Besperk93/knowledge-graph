# python extractInfo/batch_process_v2.py

import os
from processors import TranscriptProcessor

dir = "./Vault/transcripts"

extractor = TranscriptProcessor()
extractor.batch_process(dir, 10)
