import info_extractor

script_location = "Vault/testMathData"
output_location = "Vault/graph"

extractor = infoExtractor.infoExtractor(script_location, output_location)
extractor.open_transcript()
extractor.display_entity_data()
