# BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval

This repository contains the code for BERT-PLI in our IJCAI-PRICAI 2020 submission: *BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval*

## Outline

### Model

- ``./model/nlp/BertPoolOutMax.py`` : model paragraph-level interactions between documents.

- ``./model/nlp/AttenRNN.py`` : aggregate paragraph-level representations. 

### Config

- ``./config/nlp/BertPoolOutMax.config`` : parameters for ``./model/nlp/BertPoolOutMax.py`` .

- ``./config/nlp/AttenGRU.config`` / ``./config/nlp/AttenLSTM.config`` : parameters for ``./model/nlp/AttenRNN.py`` (GRU / LSTM, repectively)


### Formatter

- ``./formatter/nlp/BertDocParaFormatter.py`` : prepare input for ``./model/nlp/BertPoolOutMax.py``
  An example:

```
{
	"guid": "queryID_docID",
	"q_paras": [...], // a list of paragraphs in query case,
	"c_paras": [...], // a list of parameters in candidate case,
	"label": 0, // 0 or 1, denote the relevance
}
```

- ``./formatter/nlp/AttenRNNFormatter.py`` : prepare input for ``./model/nlp/AttenRNN.py``
  An example:

```
{
	"guid": "queryID_docID",
	"res": [[],...,[]], // N * 768, result of BertPoolOutMax,
	 "label": 0, // 0 or 1, denote the relevance
}
```

### Scripts

- ``poolout.py``/``train.py``/``test.py``, main entrance for *poolling out*, *training*, and *testing*.

### Requirements

- See ``requirements.txt``

## How to Run?

- Get paragraph-level interactions by BERT: 

```bash
python3 poolout.py -c config/nlp/BertPoolOutMax.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
```

- Train

```bash
python3 train.py -c config/nlp/AttenGRU.config -g [GPU_LIST] 
```

or 

```bash
python3 train.py -c config/nlp/AttenLSTM.config -g [GPU_LIST] 
```

- Test

```bash
python3 test.py -c config/nlp/AttenGRU.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
```

or 

```bash
python3 test.py -c config/nlp/AttenLSTM.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
```

## Data

Please refer to [COLIEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/)

## Note 

Now the eval_metric are specific to the COLIEE task, please stay tuned for updates.

## Contact
For more details, please refer to our paper **BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval** (*To Appear*). If you have any questions, please email shaoyq18@mails.tsinghua.edu.cn . 
