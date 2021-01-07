Where we seek guidance:
> We have brainstormed ideas (see below) on where to put our efforts. What are realistic / achievable expansions that are worth pursuing?
> Does it make sense to participate in ML Repro Challenge given that codebase is given?

# Basic approach
1. get code running and try to replicate results (code base alreay exists & we have it running, should we re-implement the code base? Or just go with it and add experiments/models/etc?)
2. if results differ, investigate & fix (we expect results will match up and there is not much interesting to do here)
3. if replicated: expand. Big question is where & how

# Our emerging narrative on expanding this work
Their central claim: attention applied to LSTM output can be made faithful/plausible by ensuring "diversity" between the cell output vectors over the sequence.

"Fundamental" Downsides of current approach -> how to improve
- If attention does not provide faithful explanations, why is attention effective? How does attention improve performance of these models?
- In our view: attention applied in this way doesn't add much value: all info is represented in last LSTM cell output. We believe this may make the result more plausible & faithful
   - smarter way of applying attention: bi-lstm, or attention on input words (type of attention and where in the model)
- Their initial problem statement: attention don't provide faithful/plausible explanations. Different explanation methods don't correlate; "if a model gives same result for different explanations (attentions), it can't be right" - is that a correct premise? 
- These ideas might be useful/applicable in other domains, we can explore image classification

"Correctness/completeness" critique on current approacch -> how to improve
- Ortho model out of the picture in 2nd half of paper; we could fill in the blanks (apply evaluations)
- Evaluation methods: correlation measures vs gradient & integrated gradient methods. But Stefan indicated: be careful with this type of comparison 
- Just looked at LSTM (title implies a much broader application), let's look at other models with attention (in the NLP domain)?

# Expansion ideas

#### FILLING IN BLANKS IN THE PAPER?
* Minimal Rational evaluation: add Ortho LSTM?
* Where is attention applied in the model, can this be designed in a smarter way?
* Zero vs. Xavier initialization of the cell state/output at t=0

#### APPLY TO MORE/OTHER DATASETS
- Explore QA domain more (unexplained results, could investigate more)
- Expanding by testing on more datasets doesn't seem very interesting: already 14 in the paper

#### EXPAND TO NEW MODEL VARIATIONS

* Can we apply the solutions offered to other models/architectures:
  - Bi-LSTM (expect better attention distribution to start with?)
  - Transformers? DistilBERT?
  - Non sequential?
  - Investigate MLP transparency?
  - ABcnn?
* Image processing? (plausibility concept very different)

#### DESIGN OTHER FIXES for EXPLAINABLE ATTENTION?

- can we think of another solution besides ortho & diversity LSTM?

#### EXPAND TO NEW TASKS

* Candidates:
  - Translation
  - Co-reference resolution?
  - Summarization?
  - Generative mode of LSTM?
* Image processing domain

#### EXPAND EVALUATION METRICS

- Other methods?
- New ways to incorporate human judgements?


# Questions about code

  - In the code, orto_lstm.py, line 85: why xavier initialize the hidden states (instead of zeros?)
  - in `configurations.py` ln 51 hidden_size=128?
    and in ln 54 generator hidden size is always 64?
    but then in `Encoder.py` ln 37 hidden_size = 2 * hidden_size
  - what is the generator class used for? which tasks require a generator?
  - h_bar is not calculated as the mean in eq1 of the paper, but in the code it is

# Questions on the content of the paper
  - Authors state that it is surprising that vanilla LSTMs put much attention to punctuation tokens. But isn't that to be expected (most complete representation of the sequence)
  - sec5.2, pp4211: how does the MLP with attention work? How does it act on input with different sequence length?
