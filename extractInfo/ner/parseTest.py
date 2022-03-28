import infoExtractor

script_location = "Vault/testScripts"
output_location = "Vault/graph"

extractor = infoExtractor.infoExtractor(script_location)
extractor.open_transcript()
extractor.display_entity_data()
