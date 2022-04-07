import info_extractor

script_location = "Vault/testScripts"
output_location = "Vault/graph"

extractor = info_extractor.infoExtractor(script_location, output_location)
extractor.open_transcript()
extractor.display_entity_data()
