from inference import InferencePipeline


extractor = InferencePipeline()

test1 = "The cat sat on the mat and licked it's paw"
preds = extractor.extract_relations(test1)
print(f"Best: {preds[0]} of {len(preds)}")

# test2 = "After eating the chicken, he developed a sore throat the next morning."
# extractor.extract_relations(test2)
# test3 = "The surprise visit caused a frenzy on the already chaotic trading floor."
# extractor.extract_relations(test3)


# test_script = "Vault/testScripts/4 Polygon perimeter - YouTube.txt"
# extractor.extract_relations(test_script)
