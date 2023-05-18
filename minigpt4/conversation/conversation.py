import argparse
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from minigpt4.common.registry import registry
import open3d as o3d
import numpy as np
import random
import math
import cv2
import copy
NUM_PICTUR=4
PICTURE_WIDTH=640
PICTURE_HEIGHT=640
r_x = [0] * NUM_PICTUR
r_y = [0] * NUM_PICTUR
r_z = [0] * NUM_PICTUR
cloud = [0] * NUM_PICTUR

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)



class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)
        try:
            print(type(conv))
            print("conv:",conv)
            print("conv.messages:",conv.messages)
        except:
            pass

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        image_embs = []
        raw_cloud = o3d.io.read_point_cloud(image.name)
        # o3d.visualization.draw_geometries([cloud])
        for i in range(NUM_PICTUR):
            r_x[i] = random_number = random.random() * 2 * math.pi
            r_y[i] = random_number = random.random() * 2 * math.pi
            r_z[i] = random_number = random.random() * 2 * math.pi
            cloud[i] = copy.deepcopy(raw_cloud)
            R = o3d.geometry.get_rotation_matrix_from_xyz((r_x[i], r_y[i], r_z[i]))  # 欧拉角转变换矩阵
            cloud[i].rotate(R, center=(0, 0, 0))
        
        # print('沿X轴旋转90度，沿Z轴旋转45度的欧拉角转换成旋转矩阵为\n', R1)
        
        # o3d.visualization.draw_geometries([cloud[0].scale(0.5,center=cloud[0].get_center()),cloud[1].translate((1,0,0)),cloud[2].translate((0,1,0)),cloud[3].translate((1,1,0))])
        
        step = 0.001  # 像素大小
        # ------------------创建像素格网-----------------------
        for i in range(NUM_PICTUR):
            # o3d.visualization.draw_geometries([cloud[0]])
        
            point_cloud = np.asarray(cloud[i].scale(0.5, center=cloud[i].get_center()).points)
            # print(point_cloud.shape)
            # print(point_cloud)
            # 1、获取点云数据边界
            x_min, y_min, z_min = np.amin(point_cloud, axis=0)
            x_max, y_max, z_max = np.amax(point_cloud, axis=0)
            print(x_min, x_max, y_min, y_max)
            # 2、计算像素格网行列数
            width = np.ceil((x_max - x_min) / step)
            height = np.ceil((y_max - y_min) / step)
            print("像素格网的大小为： {} x {}".format(width, height))
            # 创建一个黑色的空白图像
            # img = np.zeros((int(width), int(height)), dtype=np.uint8)
            img = np.zeros((int(PICTURE_WIDTH), int(PICTURE_HEIGHT)), dtype=np.uint8)
        
            img.fill(255)  # 设置图片背景颜色，默认为：黑色。
            # print(img)
            # 3、计算每个点的像素格网索引，并将有点的像素格网赋值为白色
            for i in range(len(point_cloud)):
                col = np.floor((point_cloud[i][0] - x_min) / step + (PICTURE_WIDTH - (x_max - x_min) / step) / 2)
                row = np.floor((point_cloud[i][1] - y_min) / step + (PICTURE_HEIGHT - (y_max - y_min) / step) / 2)
                img[int(col), int(row)] = 0
            # 生成的图片与实际视角偏差90°，因此做一下旋转
            img90 = np.rot90(img)
            rgb_image = cv2.cvtColor(img90, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
      
            # image=Image.open("/content/1a04e3eab45ca15dd86060f189eb133.png")
            print("open suceess")
            raw_image = img_pil
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            image_emb, _ = self.model.encode_img(image)
            image_embs.append(image_emb)
        
        # image_embs = torch.stack(image_embs)

        sum_image_emb = torch.stack(image_embs).sum(dim=0)
        avg_image_emb = sum_image_emb / len(image_embs)

        img_list.append(avg_image_emb)
        # img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        print("done")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs


