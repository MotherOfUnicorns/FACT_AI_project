- Frank check whether first part is going well

- Bespreek dit stukje uit paper, misschien link leggende met Frank resultaat of met orthoQfix:
In tasks such as paraphrase detection, the model
is naturally required to carefully go through the
entire sentence to make a decision and thereby re-
sulting in delayed decision flips. In the QA task,
the attention ranking in the vanilla LSTM model
itself achieves a quick decision flip. On further in-
spection, we found that this is because these models
tend to attend onto answer words which are usually
a span in the input passage. So, when the repre-
sentations corresponding to the answer words are
erased, the model can no longer accurately predict
the answer resulting in a decision flip.

- discuss extensions: 
  - *ortho Qfix* and what we expect from experiment reruns using this fix (how can it be that performance collapses on SST, but not on other tasks where ortho is applied in the Qencoder?)
  - LIME, the scores are not context-aware, and does not depend on the position of the words (e.g. Sandra went to the *garden*, Daniel went to the *garden*.)
  - other attentions (any thoughts on what could be interesting? Maybe Multi-Head Attention IS comparable?), 
  - other base models (BiLSTM: hidden state combineren of gewoon concat? Als concat hoe om te gaan met hogere aantal dims)


- how to reproduce Random Conicity Table 2? (Yun's fisrt attempt)

- what is exactly shown in the violin plots? (what is on the x-axis in the QA tasks?)
