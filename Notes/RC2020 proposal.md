# Basics
Run the code:
- does it work? what changes did we need to make?
- do we get the same results?

# Robustness
Add results for orthogonality variants
Look at effect of initialisation (with zero's)
Check impact of multiple seeds on: conicity, accuracy
Are differences significant --> use signtest

DO WE AGREE WITH THEIR CLAIMS?

# Review the definitions
- what is faithful
- what is plausible
- how do they relate (is there ONE best explanation for a prediction? what combinations can theoretically exist, for example can we have non-faithful explanations that are plausible)
- look at not not paper
- Are their proposed metrics quantifying the concepts of faithfulness and plausibility well?

# Checking assumptions, expectation and hypotheses
How can attention work at all if all hidden states have narrow conicity  -> maybe attention isn't doing its job in these tasks
Side question: Is the relation that the authors claim to exist between conicity and faithfulness correct?

Is attention improving performance in the tasks they researched: sentiment analysis, Natural language inference, paraphrase detection, Q&A [all seq-to-one tasks]
- in their implementation 
- in other papers?
Look at literature about LSTM
- for which task DOES it work, and add value?

OUR MAIN POINT/HYPOTHESIS
Frank:
- Differences in attention weights ONLY have a meaning (faithfulness + plausibility) when attention works in itself.
- If adding attention to a model does NOT lead to improved performance, then you should not expect that attention weights are a faithfull and plausible explanation of the predictions
- If attention does NOT add value, then it makes sense to diversity + orthogonal approach

Pieter:
A necessary (but possibly not sufficient) condition for an attention mechanism to provide faithful and plausible explanations is that the attention mechanism is actually ‘working’. An attention mechanism ‘works’ when it contributes to model performance (e.g. increases accuracy). It is not reasonable to expect that an attention mechanism that doesn’t do useful processing will provide meaningful explanations. In this research we verify whether this condition is satisfied in the experiments conducted in (author name et al., 2020). We will show this condition is not satisfied in their experiments, and show that in training regimes where presence of an attention mechanism is contributing to performance, the explanations it provides are faithful (and plausible).

Yun:
Attention can provide faithful(/plausible) explanations only when the addition of the attention mechanism changes the dynamics of the underlying model (in this case, LSTM).
If we observe improved accuracy with the addition of attention, then we can conclude that it's indeed working its magic and producing faithful explanations, but the inverse is not necessarity true.
In other words, if the attention mechanism is not contributing to the model's learning process (which may manifest as unchanged accuracy), we cannot expect that the attention weights provide a meaningful explanation.
Diversifying the LSTM outputs is then only an attempt to force the model dynamics into the case where attention plays a role, but we question whether this is a good remedy.

Follow-up question: Does the diversity and/or orthogonality variant also improve faithfullness of explanations in these cases? [CAN it make it better?]

# The way we are going to check this:
Analyze a task + model where attention works
- different tasks (where attention has impact), for example coreference resolution
- other attention mechanism in the LSTM
- attention in other type of model: Bi-LSTM, ConvNet, ...

Look at:
- accuracy: does it indeed work?
- conicity: are values lower (-> wider conicity) in task where attention has significant impact on performance?
- weights: are weights faithfull and plausible?



