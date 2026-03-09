


git clone https://huggingface.co/spaces/ASLP-lab/DiffRhythm2
或者
hf download ASLP-lab/DiffRhythm2 --repo-type=space --local-dir ./DiffRhythm2


conda create -n diffrhythm2 python=3.10 -y   

conda activate diffrhythm2


cd diffrhythm2

 pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

 pip install -r requirements.txt 


pip install gradio   

pip install spaces    


pip install onnxruntime-gpu    

 python app.py

出现错误UnicodeDecodeError: 'gbk' codec can't decode byte 0xbc in position 955: illegal multibyte sequence
Python的json.load()在读取文件时，使用了系统默认的‘gbk’编码，而你要读取的词典文件（vocab）的实际编码是‘utf-8’，导致解码失败。
with open(vocab_path, "r") as f:
修改为
with open(vocab_path, "r", encoding='utf-8') as f:

"C:\Users\jybwo\DiffRhythm2\g2p\g2p_generation.py"
"C:\Users\jybwo\DiffRhythm2\g2p\g2p\__init__.py"
"C:\Users\jybwo\DiffRhythm2\g2p\utils\g2p.py"
"C:\Users\jybwo\DiffRhythm2\diffrhythm2\utils.py"









谢谢你......我还为Discord频道上一个遇到很大困难的人做了一个总结，说明我需要在Windows上运行所需的条件。在Ubuntu上运行起来容易多了......(!)

创建Windows本地服务器us.txt

使用本地GPU创建Windows本地服务器 https://huggingface.co/spaces/ASLP-lab/DiffRhythm

创建一个Python 3.12.9 venv
==============================
下载并安装Python 3.12.9，使用Add PIP和Add PythonPath选项
。可在Python找到，https://www.python.org/downloads/windows/
Open Powershell
py -3.12 -m venv diff-gradio
python -m venv diff-gradio
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -scope CurrentUser
diff-gradio\Scripts\activate

克隆Hugging Face仓库
==============================
从
https://git-scm.com/downloads/win git LFS下载
并安装git，安装git，https://huggingface.co/spaces/ASLP-lab/DiffRhythm
cd：\Users\44741\DiffRhythm，<---需要切换到本地路径
，还没安装requirements.txt

Windows安装库
================================
a） Triton
Python --版本
从
https://huggingface.co/madbuda/triton-windows-builds/tree/main PIP下载正确版本的Triton.WHL 安装 C：\Users\44741\DiffRhythm\triton-3.0.0-cp312-cp312-win_amd64.whl <---需要切换到本地路径
b） pytorch使用cuda
PIP 卸载Torch
Pip 缓存清除
nvidia-smi（了解你的CUDA版本）
进入 https://pytorch.org/get-started/locally/ 获取PIP安装（但命令中的PIP3改为PIP），
例如pip install torch、torchvision、torchaudio、--index-url -c，https://download.pytorch.org/whl/cu118
c） requirements.txt
c：\Users\44741\DiffRhythm <---需要切换到本地路径
pip install -r requirements.txt
d） CUDA 版本的 OnnyXRuntime
PIP 安装 OnNXryn-GPU

对克隆仓库的文件进行修改，使其在Windows上正常运行
=======================================================================
文件名文件夹
init.py diffrhythm\g2p\g2p
g2p.py diffrhythm\g2p\utils
g2p_generation.py diffrhythm\g2p
infer_utils.py diffrhythm\infer
app.py 根

diffrhythm\g2p\g2p_init_.py，
第28行，open（vacab_path，“r”，encoding=“utf-8”）为f：

diffrhythm\g2p\utils\g2p.py，第63行
，开启（“./diffrhythm/g2p/utils/mls_en.json”，“r”，编码=“UTF-8”）为F：

diffrhythm\g2p\g2p_generation.py，第117行
，开启（“./diffrhythm/g2p/g2p/vocab.json”，“r”，编码=“UTF-8”）为F：

DIFFRHYTHM\INFER\infer_utils.py，第105行
，文件为open（'./diffrhythm/g2p/g2p/vocab.json'，'r'，编码='UTF-8'）：

app.py，在第12和第33行注释，在第11
行12 #import
33 #@spaces后添加环境变量。GPU（duration=20）
12 os.environ[“PHONEMIZER_ESPEAK_PATH”] = r“C：\Program Files\eSpeak NG”
13 os.environ[“PHONEMIZER_ESPEAK_LIBRARY”] = r“C：\Program Files\eSpeak NG\libespeak-ng.dll”

安装 espeak-NG
即可
====================查看 https://www.youtube.com/watch?v=J8FejpiGcAU
a） 下载并安装 eSpeak NG
https://github.com/espeak-ng/espeak-ng/releases

b） 安装 MbrolaTools35
https://archive.org/details/MbrolaTools35

c） 下载额外文件
https://github.com/thiekus/MBROLA/releases/download/3.3/mbrola_build_3.3_rev2.zip
复制所有文件，来自 mbrola_build_3.3_rev2\Win64\Release
Paste 的 C：\Program Files\eSpeak NG 内

d） 下载MBROLA配音 https://github.com/numediart/MBROLA-voices

下载jp1， jp2， jp3
C：\Program Files\eSpeak NG\espeak-ng-data\mbrola\

e） 下载 MBROLA 语音
https://github.com/user-attachments/files/19094349/mbrola_ph.zip
C：\Program Files\eSpeak NG\espeak-ng-data

在 Venv
CD 中运行 Gradio 服务器
========================
C：\Users\44741\DiffRhythm <---需要切换到本地路径
Python app.py


