import gradio as gr
import torch
import random
import numpy as np
import base64
import os
from pathlib import Path
# 移除 spaces 导入
# import spaces

from diffrhythm2.utils import (
    prepare_model,
    parse_lyrics,
    get_audio_prompt,
    get_text_prompt,
    inference,
    inference_stream
)

# ========== 本地部署关键配置 ==========
# 1. 指定设备为本地CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 2. 根据你的GPU能力选择精度。如果显存不足 (如 < 8GB)，可改为 torch.float32
dtype = torch.float16 if device == 'cuda' else torch.float32

print(f"📱 运行设备: {device}")
print(f"🔧 计算精度: {dtype}")
if device == 'cuda':
    print(f"🎮 GPU型号: {torch.cuda.get_device_name(0)}")

# ========== 智能模型路径选择 ==========
def get_smart_model_path():
    """智能选择模型路径：优先本地，后在线"""
    ckpt_dir = Path("./ckpt")
    
    # 检查本地模型文件是否完整
    required_files = ["model.safetensors", "model.json", "decoder.bin", "decoder.json"]
    local_files_exist = all((ckpt_dir / f).exists() for f in required_files)
    
    if local_files_exist:
        print("🔍 检测到本地模型文件，使用本地模型...")
        for file_name in required_files:
            file_path = ckpt_dir / file_name
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   📄 {file_name} ({size_mb:.1f} MB)")
        return str(ckpt_dir)  # 返回本地目录路径
    else:
        print("🌐 未找到完整的本地模型，使用在线模型...")
        print("   注意：首次运行会自动下载模型到 ckpt/ 目录")
        return "ASLP-Lab/DiffRhythm2"  # 返回在线仓库名

# 获取模型路径
model_path = get_smart_model_path()
print(f"📂 模型路径: {model_path}")

# 提前加载模型 (本地部署的核心)
print("⏳ 正在加载DiffRhythm2模型，请稍候...")
# 注意：prepare_model函数会检查本地文件，如果文件存在就直接使用，不存在则下载
diffrhythm2, mulan, lrc_tokenizer, decoder = prepare_model(model_path, device, dtype)
print("✅ 模型加载完成！")
# ======================================

MAX_SEED = np.iinfo(np.int32).max

# 移除 @spaces.GPU 装饰器，直接使用本地GPU
def infer_music(
        lrc, 
        current_prompt_type,
        audio_prompt=None, 
        text_prompt=None, 
        seed=42, 
        randomize_seed=False, 
        steps=16, 
        cfg_strength=1.0, 
        file_type='wav', 
        odeint_method='euler',
    ):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    torch.manual_seed(seed)
    print(f"🎲 使用随机种子: {seed}, 提示类型: {current_prompt_type}")
    try:
        lrc_prompt = parse_lyrics(lrc_tokenizer, lrc)
        lrc_prompt = torch.tensor(sum(lrc_prompt, []), dtype=torch.long, device=device)
        if current_prompt_type == "audio":
            style_prompt = get_audio_prompt(mulan, audio_prompt, device, dtype)
        else:
            style_prompt = get_text_prompt(mulan, text_prompt, device, dtype)
    except Exception as e:
        raise gr.Error(f"❌ 处理输入时出错: {str(e)}")
    
    style_prompt = style_prompt.to(dtype)
    print("🎵 开始音乐生成...")
    generate_song = inference(
        model=diffrhythm2, 
        decoder=decoder, 
        text=lrc_prompt, 
        style_prompt=style_prompt,
        sample_steps=steps,
        cfg_strength=cfg_strength,
        odeint_method=odeint_method,
        duration=240,
        file_type=file_type
    )
    print("✅ 生成完成！")
    return generate_song

css = """
/* 固定文本域高度并强制滚动条 */
.lyrics-scroll-box textarea {
    height: 405px !important;
    max-height: 500px !important;
    overflow-y: auto !important;
    white-space: pre-wrap;
    line-height: 1.5;
}
.gr-examples {
    background: transparent !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px;
    margin: 1rem 0 !important;
    padding: 1rem !important;
}
"""

def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

with gr.Blocks(css=css) as demo:
    # 根据模型路径显示不同的标题
    if os.path.isdir(model_path):
        title_html = f"""
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
                    Di♪♪Rhythm 2 (谛韵) - 本地模型版
                </div>
                <div style="display:flex; justify-content: center; column-gap:4px;">
                    <a href="https://arxiv.org/pdf/2510.22950">
                        <img src='https://img.shields.io/badge/Arxiv-Paper-blue'>
                    </a> 
                    <a href="https://github.com/ASLP-lab/DiffRhythm2">
                        <img src='https://img.shields.io/badge/GitHub-Repo-green'>
                    </a> 
                    <a href="https://aslp-lab.github.io/DiffRhythm2.github.io/">
                        <img src='https://img.shields.io/badge/Project-Page-brown'>
                    </a>
                </div>
                <div style="margin-top: 10px; color: green; font-weight: bold;">
                    ✅ 本地模型 | 🚀 无需下载 | ⚡ 快速加载
                </div>
            </div> 
        """
    else:
        title_html = f"""
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
                    Di♪♪Rhythm 2 (谛韵) - 在线模型版
                </div>
                <div style="display:flex; justify-content: center; column-gap:4px;">
                    <a href="https://arxiv.org/pdf/2510.22950">
                        <img src='https://img.shields.io/badge/Arxiv-Paper-blue'>
                    </a> 
                    <a href="https://github.com/ASLP-lab/DiffRhythm2">
                        <img src='https://img.shields.io/badge/GitHub-Repo-green'>
                    </a> 
                    <a href="https://aslp-lab.github.io/DiffRhythm2.github.io/">
                        <img src='https://img.shields.io/badge/Project-Page-brown'>
                    </a>
                </div>
                <div style="margin-top: 10px; color: orange; font-weight: bold;">
                    🌐 在线模型 | 📥 首次运行需下载
                </div>
            </div> 
        """
    
    gr.HTML(title_html)
    
    with gr.Tabs() as tabs:
        
        # page 1
        with gr.Tab("Music Generate", id=0):
            with gr.Row():
                with gr.Column():
                    lrc = gr.Textbox(
                        label="Lyrics",
                        placeholder="Input the full lyrics",
                        lines=12,
                        max_lines=50,
                        elem_classes="lyrics-scroll-box",
                        value="""[start]
[intro]
[verse]
Thought I heard your voice yesterday
When I turned around to say
That I loved you baby
I realize it was juss my mind
Played tricks on me
And it seems colder lately at night
And I try to sleep with the lights on
Every time the phone rings
I pray to God it's you
And I just can't believe
That we're through
[chorus]
I miss you
There's no other way to say it
And I can't deny it
I miss you
It's so easy to see
I miss you and me
[verse]
Is it turning over this time
Have we really changed our minds about each other's love
All the feelings that we used to share
I refuse to believe
That you don't care
[chorus]
I miss you
There's no other way to say it
And I and I can't deny it
I miss you
[verse]
It's so easy to see
I've got to gather myself as together
I've been through worst kinds of weather
If it's over now
[outro]"""
                    )
                    current_prompt_type = gr.State(value="text")
                    with gr.Tabs() as inside_tabs:
                        with gr.Tab("Text Prompt"):
                            text_prompt = gr.Textbox(
                            label="Text Prompt",
                            value="Pop, Piano, Bass, Drums, Happy",
                            placeholder="Enter the Text Prompt, eg: emotional piano pop",
                        )
                        with gr.Tab("Audio Prompt"):
                            audio_prompt = gr.Audio(label="Audio Prompt", type="filepath")
                        
                        def update_prompt_type(evt: gr.SelectData):
                            return "text" if evt.index == 0 else "audio"

                        inside_tabs.select(
                            fn=update_prompt_type,
                            outputs=current_prompt_type
                        )
                    
                with gr.Column():
                    
                    with gr.Accordion("Best Practices Guide", open=True):
                        gr.Markdown(f"""
                        **模型状态**: {'✅ 本地模型' if os.path.isdir(model_path) else '🌐 在线模型'}
                        
                        1. **Lyrics Format Requirements**
                        - Each line must follow: `Lyric content`
                        - Example of valid format:
                            ``` 
                            [intro]
                            [verse]
                            Thought I heard your voice yesterday
                            When I turned around to say
                            ```

                        2. **Audio Prompt Requirements**
                        - Reference audio should be ≥ 1 second, Audio >10 seconds will be randomly clipped into 10 seconds
                        - For optimal results, the 10-second clips should be carefully selected
                        - Shorter clips may lead to incoherent generation
                        
                        3. **Supported Languages**
                        - Chinese and English

                        **Due to issues with Gradio's streaming audio output, we will update the streaming feature in the future. Please stay tuned!**
                        """)
                    lyrics_btn = gr.Button("Generate", variant="primary")
                    audio_output = gr.Audio(label="Audio Result", elem_id="audio_output")
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=0,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)      
                        
                        steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=16,
                            step=1,
                            label="Diffusion Steps",
                            interactive=True,
                            elem_id="step_slider"
                        )
                        cfg_strength = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1.3,
                            step=0.5,
                            label="CFG Strength",
                            interactive=True,
                            elem_id="step_slider"
                        )
                        
                        odeint_method = gr.Radio(["euler", "midpoint", "rk4","implicit_adams"], label="ODE Solver", value="euler")                        
                        file_type = gr.Dropdown(["wav", "mp3", "ogg"], label="Output Format", value="mp3")

    tabs.select(
    lambda s: None, 
    None, 
    None 
    )
    
    lyrics_btn.click(
        fn=infer_music,
        inputs=[
            lrc, 
            current_prompt_type,
            audio_prompt, 
            text_prompt, 
            seed, 
            randomize_seed, 
            steps, 
            cfg_strength, 
            file_type, 
            odeint_method,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    # 本地运行，可以设置 server_name 和 server_port
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
