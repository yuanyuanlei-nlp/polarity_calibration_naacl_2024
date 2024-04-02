# Read Me

**Paper:** Polarity Calibration for Opinion Summarization<br/>
**Accepted:** The 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)<br/>
**Authors:** Yuanyuan Lei, Kaiqiang Song, Sangwoo Cho, Xiaoyang Wang, Ruihong Huang, Dong Yu<br/>
**Paper Link:** paper link coming soon


<br/>

## Task Description

The paper works on opinion summarization task. Opinion summarization enables automatically generating a summary from a large volume of opinions, reviews, or subjective text. The challenge of opinion summarization lies in presenting diverse or even conflicting opinions. This paper experiments on two types of opinion summarization tasks: summarizing product reviews that contain both positive and negative opinions, and summarizing multiple political articles that hold liberal or conservative political stances.


<br/>

## Dataset Description

We evaluate our approach using two datasets, each focusing on different types of opinions:

* **AmaSum**: the Amazon product reviews summarization dataset (https://github.com/abrazinskas/SelSum/tree/master/data). We use the version that contains a maximum of 100 reviews per product for experiments.

* **NeuS:** the political opinions articles summarization dataset (https://github.com/HLTCHKUST/framing-bias-metric/tree/main/data). The articles with different political stances that discuss the same event are grouped together as a cluster.



<br/>

## Model Generated Summaries

We release the generated summaries from different models:

* **generated_summary_AmaSum**: the generated summaries on AmaSum dataset
  * _test_file_names:_ the list of test file names, the model generated summaries are listed in the same order as this list
  * _base_summarizer_flan_t5_large:_ the generated summaries from our base summarizer
  * _calibrated_summarizer_PoCa:_ the generated summaries from our calibrated summarizer after polarity calibration (PoCa)
  * _lexrank:_ the generated summaries from LexRank model
  * _copycat:_ the generated summaries from CopyCat model
  * _bimeanave_avg:_ the generated summaries from BiMeanVAE-average model
  * _bimeanave_coop:_ the generated summaries from BiMeanVAE-COOP model
  * _qt:_ the generated summaries from QT model
  * _semae:_ the generated summaries from SemAE model
  * _hercules_abstractive:_ the generated summaries from Hercules abstractive summarization model
  * _hercules_extractive:_ the generated summaries from Hercules extractive summarization model
  * _gpt_35_turbo:_ the generated summaries from gpt-3.5-turbo model
  * _gpt_4:_ the generated summaries from gpt-4 model

* **generated_summary_NeuS:** the generated summaries on NeuS dataset
  * _test_data:_ the test set, the model generated summaries are listed in the order of test_0, test_1, ..., test_306
  * _base_summarizer_flan_t5_large:_ the generated summaries from our base summarizer
  * _calibrated_summarizer_PoCa:_ the generated summaries from our calibrated summarizer after polarity calibration (PoCa)
  * _lexrank:_ the generated summaries from LexRank model
  * _bart_large:_ the generated summaries from BART-large baseline
  * _pegasus_large:_ the generated summaries from Pegasus-large baseline
  * _gpt_35_turbo:_ the generated summaries from gpt-3.5-turbo model
  * _gpt_4:_ the generated summaries from gpt-4 model


<br/>

## Code Description

* **base_summarizer_AmaSum:** the code for training our base summarizer on AmaSum dataset
* **base_summarizer_NeuS:** the code for training our base summarizer on NeuS dataset
* **polarity_calibration_AmaSum:** the code for training our calibrated summarizer (PoCa) on AmaSum dataset
* **polarity_calibration_NeuS:** the code for training our calibrated summarizer (PoCa) on NeuS dataset


<br/>


## Citation

If you are going to cite this paper, please use the form:

Yuanyuan Lei, Kaiqiang Song, Sangwoo Cho, Xiaoyang Wang, Ruihong Huang, and Dong Yu. 2024. Polarity Calibration for Opinion Summarization. In Proceedings of the 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Mexico City, Mexico. Association for Computational Linguistics.


```bibtex


```







