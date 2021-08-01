# Shirin-Sokhan

A small project to finetune a pretrained gpt2 on poem dataset.

## Dataset
We crawled the awesome [Ganjoor](https://ganjoor.net/) website. Then we cleaned the text to include a standard Persian character set only. Then we added poet name as a token at the beginning of each poem. Visit [data preparation notebook](data_preparation.ipynb) for more detail on data preprocessing and [data module](src/data.py) for dataset implementation.

## Model
We used the pretrained Persian GPT2 accessible from [here](https://huggingface.co/HooshvareLab/gpt2-fa). We used this model's tokenizer with added special tokens.
Visit [model module](src/model.py) for more details of implementation using `transformers` package.

## Training 
We used `pytorch lightning` as backbone. Visit [main notebook](main.ipynb) for more details on training and generation.

## Demo
We used `streamlit` to create an app for demo. run `streamlit run demo.py`

## References
[1] [https://github.com/hooshvare/parsgpt](https://github.com/hooshvare/parsgpt)
