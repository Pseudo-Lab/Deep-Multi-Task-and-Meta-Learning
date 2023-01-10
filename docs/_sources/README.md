# Deep-Multi-Task-and-Meta-Learning

## 디펜던시 설치하기

```bash
pip install -r requirements.txt
```

## 포스트 추가 및 페이지 빌드

* `_posts` 디렉토리에 각 주제별로 폴더가 분리되어 있습니다. 
  * e.g. `_posts/continual-learning`, `_posts/multitask-meta-learning`. 
* 해당 주제에 맞는 자료를 마크다운(`*.md`)이나 Ipython 노트북(`*.ipynb`)으로 준비해서 업로드해주시면 됩니다.
* 업로드가 완료되면 html형태로 페이지를 빌드해야 하는데, 아래의 스크립트들 중 하나를 입력하시면 됩니다.

```bash
# 해당 스크립트로 빌드하실 경우, /docs 디렉토리에 완성된 페이지가 빌드됩니다.
python build.py 
```

```bash
# 해당 스크립트로 빌드하실 경우. /_build/html 디렉토리에 완성된 페이지가 빌드됩니다.
jb build .
```
