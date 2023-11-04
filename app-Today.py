CUDA_VISIBLE_DEVICES=0
import json
import gradio as gr
from pingpong import PingPong
from pingpong.gradio import GradioAlpacaChatPPManager
import time
import os
from transformers import AutoTokenizer
import torch
import sys
sys.path.append("../../")
from component.utils import ModelUtils

STYLE = """
.custom-btn {
    border: none !important;
    background: none !important;
    box-shadow: none !important;
    display: block !important;
    text-align: left !important;
}
.custom-btn:hover {
    background: rgb(243 244 246) !important;
}

.custom-btn-highlight {
    border: none !important;
    background: rgb(151, 172, 167) !important;
    box-shadow: none !important;
    display: block !important;
    text-align: left !important;
}

#prompt-txt > label > span {
    display: none !important;
}
#prompt-txt > label > textarea {
    border: transparent;
    box-shadow: none;
}
#chatbot {
    height: 600px!important; 
    overflow: auto;
    # box-shadow: none !important;
    # border: none !important;
}
#chatbot > .wrap {
    max-height: 780px;
}
#chatbot + div {
  border-radius: 35px !important;
  width: 80% !important;
  margin: auto !important;  
}

#left-pane {
    background-color: #f9fafb;
    border-radius: 15px;
    padding: 10px;
}

#left-top {
    padding-left: 10px;
    padding-right: 10px;
    text-align: center;
    font-weight: bold;
    font-size: large;    
}

#chat-history-accordion {
    background: transparent;
    border: 0.8px !important;  
}

#right-pane {
  margin-left: 20px;
  margin-right: 70px;
}

#initial-popup {
    z-index: 100;
    position: absolute;
    width: 50%;
    top: 50%;
    height: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 35px;
    padding: 15px;
}

#initial-popup-title {
    text-align: center;
    font-size: 18px;
    font-weight: bold;    
}

#initial-popup-left-pane {
    min-width: 150px !important;
}

#initial-popup-right-pane {
    text-align: right;
}

.example-btn {
    padding-top: 20px !important;
    padding-bottom: 20px !important;
    padding-left: 5px !important;
    padding-right: 5px !important;
    background: linear-gradient(to bottom right, #f7faff, #ffffff) !important;
    box-shadow: none !important;
    border-radius: 20px !important;
}

.example-btn:hover {
    box-shadow: 0.3px 0.3px 0.3px gray !important;
}

#example-title {
  margin-bottom: 15px;
}

#aux-btns-popup {
    z-index: 200;
    position: absolute !important;
    bottom: 75px !important;
    right: 15px !important;
}

#aux-btns-popup > div {
    flex-wrap: nowrap;
    width: auto;
    margin: auto;  
}

.aux-btn {
    height: 30px !important;
    flex-wrap: initial !important;
    flex: none !important;
    min-width: min(100px,100%) !important;
    font-weight: unset !important;
    font-size: 10pt !important;

    background: linear-gradient(to bottom right, #f7faff, #ffffff) !important;
    box-shadow: none !important;
    border-radius: 20px !important;    
}

.aux-btn:hover {
    box-shadow: 0.3px 0.3px 0.3px gray !important;
}
"""

get_local_storage = """
function() {
  globalThis.setStorage = (key, value)=>{
    localStorage.setItem(key, JSON.stringify(value));
  }
  globalThis.getStorage = (key, value)=>{
    return JSON.parse(localStorage.getItem(key));
  }

  var local_data = getStorage('local_data');
  var history = [];

  if(local_data) {
    local_data[0].pingpongs.forEach(element =>{ 
      history.push([element.ping, element.pong]);
    });
  }
  else {
    local_data = [];
    for (let step = 0; step < 10; step++) {
      local_data.push({'ctx': '', 'pingpongs':[]});
    }
    setStorage('local_data', local_data);
  }

  if(history.length == 0) {
    document.querySelector("#initial-popup").classList.remove('hide');
  }
  
  return [history, local_data];
}
"""

update_left_btns_state = """
(v)=>{
  console.log(v)
  document.querySelector('.custom-btn-highlight').classList.add('custom-btn');
  document.querySelector('.custom-btn-highlight').classList.remove('custom-btn-highlight');

  const elements = document.querySelectorAll(".custom-btn");

  for(var i=0; i < elements.length; i++) {
    const element = elements[i];
    console.log(element.textContent)
    if(element.textContent.trim() == v) {
      console.log(v);
      element.classList.add('custom-btn-highlight');
      element.classList.remove('custom-btn');
      break;
    }
  }
}""" 

channels = [
    "1st Channel",
    "2nd Channel",
    "3rd Channel",
    "4th Channel",
    "5th Channel",
    "6th Channel",
    "7th Channel",
    "8th Channel",
    "9th Channel",
    "10th Channel"
]
channel_btns = []

examples = [
    "你是谁", 
    "你能做什么", 
    "你叫什么名字"
]
ex_btns = []

model_name_or_path = '/home/sda/xuguangtao/Firefly-master/Qwen/Qwen-7B'
adapter_name_or_path = '/home/sda/xuguangtao/Firefly-master/output_2/firefly-qwen-7b/checkpoint-8705'

# 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
load_in_4bit = False
# 生成超参配置
max_new_tokens = 500
history_max_len = 1000
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0
device = 'cuda'
# 加载模型
model = ModelUtils.load_model(
    model_name_or_path,
    load_in_4bit=load_in_4bit,
    adapter_name_or_path=adapter_name_or_path
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)
# QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
if tokenizer.__class__.__name__ == 'QWenTokenizer':
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.bos_token_id = tokenizer.eod_id
    tokenizer.eos_token_id = tokenizer.eod_id

# # 记录所有历史记录
# if model.config.model_type != 'chatglm':
#     history_token_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
# else:
#     history_token_ids = torch.tensor([[]], dtype=torch.long)
history_len_except_last = 0 # 记录除去最后一轮的对话的历史记录的长度，为了方便重新生成最后一轮对话
def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(query,task_history):
    input_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids
    # print(input_ids)
    eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
    user_input_ids = torch.concat([input_ids, eos_token_id], dim=1)
    all_history_token_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
    for history_chat in task_history:
        history_input, history_answer = history_chat.ping,history_chat.pong
        history_input_ids = tokenizer(history_input, return_tensors="pt", add_special_tokens=False).input_ids
        history_answer_ids = tokenizer(history_answer, return_tensors="pt", add_special_tokens=False).input_ids
        # history_input_ids = torch.concat((history_input_ids,eos_token_id),dim=1)
        # history_answer_ids = torch.concat((history_answer_ids,eos_token_id),dim=1)
        all_history_token_ids = torch.concat((all_history_token_ids,history_input_ids,eos_token_id,history_answer_ids,eos_token_id), dim=1)
    all_history_token_ids = torch.concat((all_history_token_ids,user_input_ids),dim=1)
    model_input_ids = all_history_token_ids[:, -history_max_len:].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_input_ids,max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
            temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
        )
    model_input_ids_len = model_input_ids.size(1)
    response_ids = outputs[:, model_input_ids_len:]
    response = tokenizer.batch_decode(response_ids)
    # print("Firefly：" + response[0].strip().replace(tokenizer.eos_token, "")+"\n")
    full_response = response[0].strip().replace(tokenizer.eos_token, "")
    return full_response



    return response
def add_pingpong(idx, ld, ping):
    res = [
      GradioAlpacaChatPPManager.from_json(json.dumps(ppm))
      for ppm in ld
    ]
    # time.sleep(2)
    ppm = res[idx]
    response = predict(ping,ppm.pingpongs)
    ppm.add_pingpong(PingPong(ping,response))
    #   ppm.add_pingpong(PingPong(ping, "dang!!!!!!!"))
    # 将instruction_txtbox清空
    # 
    # res的值字符串化再传给local_data
    return "", ppm.build_uis(), str(res)

def clean_pingpong(idx,ld):
    res = [
        GradioAlpacaChatPPManager.from_json(json.dumps(ppm))
        for ppm in ld
    ]

    ppm = res[idx]
    ppm.pingpongs=[]
    print(str(res))
    return ppm.build_uis(), str(res)

def regenerate_pingpong(idx,ld):
    res = [GradioAlpacaChatPPManager.from_json(json.dumps(ppm))
        for ppm in ld]
    
    ppm =res[idx]
    last_pingpong = ppm.pop_pingpong()
    last_ping = last_pingpong.ping
    response = predict(last_ping,ppm.pingpongs)
    # ppm.replace_last_pong(response)
    ppm.add_pingpong(PingPong(last_ping,response))
    return ppm.build_uis(), str(res)

def set_example(btn):
    return btn, gr.update(visible=False)

def set_popup_visibility(ld, example_block):
    return example_block

with gr.Blocks(css=STYLE, elem_id='container-col') as block:
    idx = gr.State(0)
    local_data = gr.JSON({},visible=False)  # {'ctx': '', 'pingpongs':[]}

    with gr.Row():
        with gr.Column(scale=1, min_width=180):
            gr.Markdown(" ", elem_id="left-top")
        
    #   with gr.Column(elem_id="left-pane"):
    #     with gr.Accordion("Histories", elem_id="chat-history-accordion"):
    #       channel_btns.append(gr.Button(channels[0], elem_classes=["custom-btn-highlight"]))

    #       for channel in channels[1:]:
    #         channel_btns.append(gr.Button(channel, elem_classes=["custom-btn"]))
        
    with gr.Column(scale=6, elem_id="right-pane"):
        with gr.Column(elem_id="initial-popup", visible=False) as example_block:
            with gr.Row(scale=1):
                with gr.Column(elem_id="initial-popup-left-pane"):
                    gr.Markdown("Today-chat", elem_id="initial-popup-title")
                    gr.Markdown("Today模型是在Qwen-7b模型基础上进行指令微调得到的模型")
                with gr.Column(elem_id="initial-popup-right-pane"):
                    gr.Markdown("欢迎您的使用！",elem_id="initial-popup-title")
                    gr.Markdown("它具有通用和生物医学领域问答对话、生物医学命名实体识别（NER）和生物医学关系抽取（RE）能力")

            with gr.Column(scale=1):
                gr.Markdown("Examples")
                with gr.Row() as text_block:
                    for example in examples:
                        ex_btns.append(gr.Button(example, elem_classes=["example-btn"]))

        with gr.Column(elem_id="aux-btns-popup", visible=True):
            with gr.Row():
                # stop = gr.Button("Stop", elem_classes=["aux-btn"])
                regenerate = gr.Button("Regenerate", elem_classes=["aux-btn"])
                clean = gr.Button("Clean", elem_classes=["aux-btn"])

        chatbot = gr.Chatbot(elem_id="chatbot",label='Today-chat',bubble_full_width=False, avatar_images=(os.path.join(os.path.dirname(__file__),"user.png"),os.path.join(os.path.dirname(__file__),"chat.png")))
        instruction_txtbox = gr.Textbox(
            placeholder="Ask anything", label="",
            elem_id="prompt-txt"
        )

#   # 给左侧的channel button 添加事件
#   for btn in channel_btns:
#     btn.click(
#       set_chatbot,
#       [btn, local_data],
#       [chatbot, idx, example_block]        
#     ).then(
#       None, btn, None, 
#       _js=update_left_btns_state        
#     )

  # example button的事件
  # 按下后，example框消失，按下的example内容进入输入框
    for btn in ex_btns:
        btn.click(
        set_example,
        [btn],
        [instruction_txtbox, example_block]  
        )

    # 输入框提交内容后
    # example框隐藏
    # 聊天记录更新
    # 保存到local storage
    instruction_txtbox.submit(
        lambda: gr.update(visible=False),
        None,
        example_block
    ).then(
        add_pingpong,
        [idx, local_data, instruction_txtbox],
        [instruction_txtbox, chatbot, local_data]
    ).then(
        None, local_data, None, 
        _js="(v)=>{ setStorage('local_data',v) }"
    )
    regenerate.click(
        regenerate_pingpong,
        [idx,local_data],
        [chatbot,local_data]
        ).then(
        None,
        local_data,
        None,
        _js="(v)=>{setStorage('local_data',v)}")
   
    clean.click(
        clean_pingpong,
        [idx,local_data],
        [chatbot,local_data]
        ).then(
        None,local_data,None,
        _js="(v)=>{setStorage('local_data',v)}")

    block.load(
        None,
        inputs=None,
        outputs=[chatbot, local_data],
        _js=get_local_storage,
    )  


block.queue().launch(debug=True,share=True)
