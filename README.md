## Why we need Temporal-dependence aware models?

####  Current methods predominantly use the average features over the trip as input. The example suggests that variations in speed profiles lead to disparities in total trip EC, despite the fact that the two trips exhibit a similar average trip speed, distance and duration.
![speed variation](https://github.com/user-attachments/assets/78286bad-c239-4511-b6a0-44d8c59afeee)

## How to consider the sequential interdependences between adjacent segments?

#### Through discretizing trips into ordered segments, EC prediction can be conceptualized as a sequential prediction task. Sequence-aware EC prediction can thus benefit from time-series models, as these models can learn the similar patterns of adjacent segments to assist the EC prediction of the current segment.
![method](https://github.com/user-attachments/assets/1acc4995-c6ce-4e49-adba-2e0f0670c50a)

##### 2021.7.16: added a  Multicollinearity_analysis.py

##### 2021.7.17: added a  residual_analysis.py
