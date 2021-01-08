### Jan 08, 2020
Yun:
- renamed the old repository (which was public because it was a fork) and made a new one with the same name -> you should all get a new email reminder, and no further changes are needed
- moved job scripts to ./project so they're under version control
- finished data pre-processing that I could do.
    * Anemia and Diabetes are not in the Transpanrency/preprocess folder
    * Tweets requires contacting the authors of the 'attention is not explanation' paper
    * CNN data is missing: the owner of the data moved
    
- changed arguments of torch functions (pack-padded_sequence) to fix deprecated issues in higher versions of torch (1.1.0 vs 1.7.0). Now compatible to Frank env (torch=1.7.0).
