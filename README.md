# SSL_KISDI

### versions
- Python 3.6.13
- CUDA 11.6
- pytorch 1.10.2
- gensim 3.8.3
- mecab_python-0.996_ko_0.9.2_msvc-cp36-cp36m-win_amd64

### how to execute `scopus_infer_module.py`
1. *clone git directory*
2. *unzip TTmodels.zip into `/data` directory*
3. open *anaconda prompt* and activate VENV contains above packages
4. type following command
```
python scopus_infer_module.py --transformer_model_path ./data --lda_model_path ./data/ldamodel
```
