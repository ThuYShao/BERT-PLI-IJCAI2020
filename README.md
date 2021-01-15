# BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval

This repository contains the code for BERT-PLI in our IJCAI-PRICAI 2020 paper: *BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval*. 

## File Outline

### Model

- ``./model/nlp/BertPoint.py``: model for Stage2: fine-tune a paragraph pair classification Task. 

- ``./model/nlp/BertPoolOutMax.py`` : model paragraph-level interactions between documents.

- ``./model/nlp/AttenRNN.py`` : aggregate paragraph-level representations. 

### Config

- ``./config/nlp/BertPoint.config`` : configuration of ``./model/nlp/BertPoint.py`` (Stage 2, fine-tune). 

- ``./config/nlp/BertPoolOutMax.config`` : configuration of ``./model/nlp/BertPoolOutMax.py`` .

- ``./config/nlp/AttenGRU.config`` / ``./config/nlp/AttenLSTM.config`` : configuration of ``./model/nlp/AttenRNN.py`` (GRU / LSTM, repectively)


### Formatter

- ``./formatter/nlp/BertPairTextFormatter.py``: prepare input for ``./model/nlp/BertPoint.py`` (Stage 2, fine-tune)

- ``./formatter/nlp/BertDocParaFormatter.py`` : prepare input for ``./model/nlp/BertPoolOutMax.py``

- ``./formatter/nlp/AttenRNNFormatter.py`` : prepare input for ``./model/nlp/AttenRNN.py`` 


### Examples

Examples of input data. Note that we cannot make the raw data public according to the memorandum we signed for the dataset. The examples here have been processed manually and differ from the true data.

- ``./examples/task2/data_sample.json``: example input for Stage 2 (fine-tune).

    The format:

```
    {
	"guid": "queryID_paraID",
	"text_a": text of the decision paragraph,
        "text_b": text of the candidate paragraph,
	"label": 0 or 1 
    }
```

- ``./examples/task1/case_para_sample.json``: example input used in ``./config/nlp/BertPoolOutMax.config``. 

    The format:

```
    {
	"guid": "queryID_docID",
	"q_paras": [...], // a list of paragraphs in query case,
	"c_paras": [...], // a list of parameters in candidate case,
	"label": 0, // 0 or 1, denote the relevance
    }
```

- ``./examples/task1/embedding_sample.json``: example input used in ``./config/nlp/AttenGRU.config`` and ``./config/nlp/AttenLSTM.config``

    The format:

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

- Stage 1: BM25 Selection:

    The BM25 score is calculated according to the standard [scoring function](https://en.wikipedia.org/wiki/Okapi_BM25). We set $k_1=1.5$, $b=0.75$. 

- Stage 2: BERT Fine-tuning:

    ```bash
    python3 train.py -c config/nlp/BertPoint.config -g [GPU_LIST]
    ```

- Stage 3: 

    Get paragraph-level interactions by BERT: 

    ```bash
    python3 poolout.py -c config/nlp/BertPoolOutMax.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
    ```
    Train

    ```bash
    python3 train.py -c config/nlp/AttenGRU.config -g [GPU_LIST] 

    python3 train.py -c config/nlp/AttenLSTM.config -g [GPU_LIST]
    ```

    Test

    ```bash
    python3 test.py -c config/nlp/AttenGRU.config -g [GPU_LIST] --checkpoint [path of model checkpoint] --result [path to save results] 

    python3 test.py -c config/nlp/AttenLSTM.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
    ```


## Experimental Settings

### Data

Please visit [COLIEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/) to apply for the whole dataset. 

Please email shaoyq18@mails.tsinghua.edu.cn for the checkpoint of fine-tuned BERT. 

### Evaluation Metric

We follow the evaluation metrics in [COLIEEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/). Note that results should be evaluated on the whole document pool (e.g., 200 candidate documents for each query case.)

$$ Precision = \fraq{# correctly retrieved cases (paragraphs) for all queries}{# retrieved cases (paragraphs) for all queries} $$

$$ Recall = \fraq{# correctly retrieved cases (paragraphs) for all queries}{# relevant cases (paragraphs) for all queries} $$ 

$$ F-measure = \fraq{2 \times Precision \times Recall}{Precision + Recall}

### Parameter Settings

Please refer to the configuration files for parameters for each step. 

For example, in Stage 2, $learning rate = 10^{-5}$, $batch size = 16$, $training epoch = 3$. In Stage 3, $N=54$, $M=40$, $learning rate=10^{-4}$, $weight_decay=10^{-6}$.


## Contact
For more details, please refer to our paper [BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval](https://www.ijcai.org/Proceedings/2020/0484.pdf). 
If you have any questions, please email shaoyq18@mails.tsinghua.edu.cn . 
