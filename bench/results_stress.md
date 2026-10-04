## Stress test (28 hard questions)

Accuracy: 26 of 28 = 93%

| kind | correct | questions |
| --- | --- | --- |
| new wording | 10 | 12 |
| typo or synonym | 3 | 3 |
| TAD id | 3 | 3 |
| bad input | 3 | 3 |
| two steps | 3 | 3 |
| general | 2 | 2 |
| out of scope | 2 | 2 |

Wrong:
- s02: no call find_tad({'gene': 'ERBB2'}); calls were [('find_tad', {'gene': 'HER2'})]
- s03: no call drug_link({'target': 'ERBB2', 'drug': 'lapatinib', 'lineage': 'Breast'}); calls were [('find_tad', {'gene': 'HER2'}), ('drug_link', {'target': 'HER2', 'drug': 'lapatinib', 'lineage': 'Breast'})]
