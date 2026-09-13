# Use Case 7: Multi-Modal RAG Pipeline

## Purpose
Test whether retrieval-augmented generation — retrieving similar past scenes/QA to use as in-context grounding — improves VLM planning-decision quality, extending the RAG concept from the earlier LLM series into the visual domain.

## Data
DriveLM-nuScenes, scoped specifically to **Planning** questions (not perception/prediction/behavior), because planning answers were judged to have the strongest cross-scene transferability — a plausible answer from a similar past scene is more likely to generalize than for other question types.

## Experiment Design
Design decisions were discussed before implementation:
- RAG retrieval was implemented using **sentence-transformers text embeddings on the planning question text** (rather than CLIP image embeddings), chosen for simplicity and because it aligns directly with planning-question semantics rather than raw visual similarity.
- Retrieval used leave-one-out similarity search over the DriveLM planning-question corpus, using the retrieved similar Q&A pairs as in-context examples for Qwen2.5-VL-7B-Instruct's response to the target frame's planning question.
- The person explicitly predicted throughout that results would likely not improve or might worsen, framing the exercise as a learning process about *why* RAG might fail, not as a performance optimization attempt.

## Results
- A significant data-quality discovery undermined the premise of the experiment: **DriveLM planning questions are fully templated**, with only **3 unique question strings across the entire dataset** (100% repetition rate). This caused sentence-embedding retrieval to **degenerate into effectively random sampling** — since nearly all questions are textually identical, similarity scores clustered at 1.000 for all frames, giving retrieval no discriminating signal.
- RAG showed only a **marginal ROUGE-L improvement of +0.027 overall**, attributable to a **format-imitation effect** (the retrieved examples show the model what a well-formed answer looks like) rather than genuine scene-relevant knowledge transfer.
- A planned follow-up (UC7b, presumably testing an alternative retrieval strategy) was judged unnecessary given the zero variance in similarity scores — there was nothing further to learn from it.

## Key Takeaway
RAG's value is entirely gated by whether the underlying corpus has enough genuine textual diversity for embedding-based retrieval to discriminate between examples. With a heavily templated question set, RAG cannot deliver its intended benefit (context-relevant retrieval) and instead reduces to weak format imitation — an important methodological lesson about verifying corpus diversity *before* building a RAG pipeline, not just DriveLM-specific.
