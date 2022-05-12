import tracemalloc
import os
from processors import TranscriptProcessor

tracemalloc.start()

dir = "./Vault/transcripts"

extractor = TranscriptProcessor()
extractor.batch_process(dir, 10)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
