# A Toy Model for Event Generation

Pin Guang Shih | Student ID: 110652012

English translation of the original 2025 ML course report. Historical claims are retained as claims of that report, not as validated findings of the current research pipeline. An audit found 135 examples, 102 Location values, 23 Weather values, and 49 NPC values, rather than the original counts below. Current held-out results are documented separately in INITIAL_RESULTS.md.

## 1. Future AI Capability: A Meaningful Possibility in Twenty Years

An interactive dynamic narrative world could understand context and generate consistent events in real time. This capability would go beyond producing sentences or images: after interpreting characters, world state, player behavior, quest structure, emotions, game rules, and physical constraints, an AI system could:

- Generate plausible events and plots.
- Let NPCs make autonomous decisions and respond.
- Maintain narrative consistency.
- Generate events in response to player input in real time.
- Preserve world state and memory across turns.

### Why This Capability Matters

1. Cultural and entertainment impact: generative narrative AI could transform digital games, film, and virtual worlds. Each player could experience a unique story rather than a fixed script.
2. Integration across domains: such a system would require language understanding, contextual reasoning, planning, visualization, character modeling, and world simulation. This is a cross-domain challenge that goes far beyond a single-modality model.
3. Greater creative scale: creators could build large, rich narrative worlds while AI fills in details and generates events, enabling new forms of interactive narrative art.

For these reasons, the original report identifies world understanding and dynamic narrative generation as an important capability that AI might achieve within twenty years.

## 2. Required Ingredients and Resources

### 2.1 Data

An AI system would need a large and diverse collection of training material:

- World-state descriptions: location, time, NPC state, and world conditions.
- Narrative events and cause-effect pairs.
- Character behavior and player interaction logs.
- Character dialogue.
- Narrative corpora such as novels, game scripts, and story nodes.

Annotations should clearly map a world state to a plausible event.

### 2.2 Tools

The original report lists multi-GPU clusters, training environments with substantial memory, and low-latency inference hardware such as edge AI for a full narrative system. For this toy project, it instead requires an ordinary laptop, a small dataset, and a lightweight character-level Transformer.

### 2.3 Hardware and Environment

The original hardware section repeats the requirements above: a full system would need multi-GPU computing, high-memory training, and low-latency inference. The simplified project can run on an ordinary laptop with a small dataset and a lightweight character-level Transformer.

### 2.4 Learning Setup

A future large-scale narrative system might combine self-supervised learning, RLHF, multi-agent reinforcement learning, and world-model learning.

This toy model instead uses supervised learning, a small handcrafted dataset described in the original report as 150 samples, and labeled world-state/event pairs. This setup meets the course requirement for a simplified model problem that can be fully implemented. The later repository audit found 135 samples.

## 3. Machine Learning Type: Supervised Learning

The project uses supervised learning for the following reasons.

### A Clear Input-Output Mapping

Each example contains a world-state input and an event-text output. Supervised learning directly models this mapping.

### Explicit Data and Target Signals

The author manually wrote the event descriptions used as supervision. The original report describes this as supporting consistent target quality; independent annotation-quality evaluation was not provided.

### Straightforward Evaluation

Generated events can be inspected for plausibility without defining a separate reward function. Such inspection is qualitative and does not replace a held-out benchmark or independent human ratings.

### A Solvable Model Problem

The model is conceptually simple and can be trained within the scope of a course project.

### Example Mapping

Input: Location: forest; Time: night; Player: injured; NPC: hunter; Weather: foggy

Output: A hunter notices the injured player and offers herbal medicine.

This example illustrates the task. It is not evidence of performance on unseen states.

## 4. Implementable Model Problem and Method

The implementable model problem maps a world state to a generated event.

### 4.1 Problem Design

The objective is to turn a world state into a contextually plausible narrative event.

- Input: a structured world description.
- Output: a natural-language narrative sentence.
- Intended evaluation: whether the event is semantically plausible, clear, and causally coherent.

### 4.2 Model Method

The project uses a small character-level Transformer containing:

- Character embeddings.
- Multi-head self-attention.
- Feed-forward layers.
- Autoregressive generation.
- An <EOS> marker intended to control output length.

The original report describes 150 handcrafted samples covering ten locations, ten weather types, ten NPC types, and several player states. These are historical descriptions; inspection of the repository instead found 135 samples, 102 locations, 23 weather values, and 49 NPC values.

The original report argues that the simplified architecture can capture patterns in event generation. The later research pipeline tests that proposition with grouped holdouts, retrieval comparisons, and conditioning diagnostics.

## 5. Experimental Results in the Original Report

### Training Result

The original report states that its loss curve decreased steadily and that the trained model produced sentences with clear structure and coherent meaning. These are historical qualitative claims without a held-out comparison in that report.

[ORIGINAL_FIGURES]

### Generated Example

The report presents a generation example and interprets it as consistent with a forest at night, an injured player, and a hunter NPC, describing the event as plausible and interactive. The original illustrations are reproduced above.

## 6. Discussion

### Lessons Described in the Original Report

- A small Transformer may learn basic conditional-generation patterns even with limited data. This was the original report's interpretation, not a conclusion established by the current held-out results.
- A toy model is a useful first step toward understanding larger narrative AI systems.
- Event generation needs explicit data formats and rules; otherwise, the model may hallucinate.

### Future Challenges

1. Add more world-state variables, such as items, quest progress, and character relationships.
2. Use larger language models to increase narrative depth.
3. Integrate generated events into a game engine and test their interactivity.
4. Explore RLHF to align narrative events with player preferences.
5. Generate multi-sentence events or multi-step storylines.

The original report concludes that the project illustrates the core idea of AI story generation and connects the toy implementation to its twenty-year vision of narrative AI.

### Translation and Research Status

The current experiment pipeline must be assessed separately. Its first full single-seed run selected epoch 14 using validation NLL and stopped at epoch 29. Test token overlap F1 was 0.1190 for the Transformer versus 0.3453 for retrieval, and the conditioning NLL gap was -0.00686. Outputs contained incomplete words and repetition. These results do not establish an advantage over retrieval or successful state conditioning.

This English edition preserves the original report's structure, example, illustrations, and historical claims while identifying discrepancies found in the repository audit. It is not a substitute for a new research paper. The complete multi-seed comparisons, denser reviewed dataset, ablations, and independent human evaluation remain unfinished. See RESEARCH_PLAN.md, INITIAL_RESULTS.md, and DENSE_DATASET.md for the current protocol and limitations.
