from inference import inference_pipeline


extractor = inference_pipeline()

test1 = "The tabby cat sat on the bamboo mat, waiting to be fed"
extractor.extract_relations(test1)
test2 = "After eating the chicken, he developed a sore throat the next morning."
extractor.extract_relations(test2)
test3 = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
extractor.extract_relations(test3)
