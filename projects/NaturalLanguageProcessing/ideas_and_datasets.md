# Ideas

- Question pairs [dataset](https://www.kaggle.com/c/quora-question-pairs/data).

Idea:  Train a model to identify similar questions. Then, quantify how important it is to take into account word order in analyzing questions. 
 
Method:  
Option 1 - Train an order-sensible (possibly RNN) method and see how the results degrade when changing word order.  
Option 2 - Compare the performance of two models in estimating question similarity. One that takes into account word order (possibly RNN) and one that does not (Bag of Words).

- Fake news [dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

Idea: To evaluate the difference (pros and cons) between using a pre-trained word embedding vs training an embedding specific for the task.  

Method: Train a text classification algorithm (fake news vs real news dataset suggested) in two different ways: (1) Using a pre-trained word embedding (trained on the same corpus with CBOW or download it already trained on another corpus) and (2) letting the network to learn the word embedding during the training for the classification task.

- Toxic wikipedia [dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

Idea: Train an algorithm to detect and categorize Wikipedia Toxic comments. Try different approaches to manage class imbalance.  

Method: Train a multiclass text classification algorithm on a Wikipedia comments dataset.

# Datasets

Pretrained machine translation from [huggingface]( https://huggingface.co/datasets?filter=task_ids:machine-translation)

https://analyticsindiamag.com/10-nlp-open-source-datasets-to-start-your-first-nlp-project/

[Beginner NLP datasets](https://analyticsindiamag.com/10-nlp-open-source-datasets-to-start-your-first-nlp-project/)

[Description Q&A](https://research.fb.com/downloads/babi/)
