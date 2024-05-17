import torch
from PIL import Image

from ...smp import *
from ...utils import DATASET_TYPE
from ..base import BaseModel


class LLaVA_Omni(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_pth,
        **kwargs,
    ):
        from omni.constants import MODEL_ZOOS
        from omni.models.eva_clip.factory import EVACLIPImageProcessorWrapper
        from omni.models.llava.configuration_llava import LlavaConfig
        from omni.models.llava.modeling_llava import LlavaForCausalLM
        from transformers import CLIPImageProcessor, LlamaTokenizer

        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_ZOOS[model_pth])
        self.config = LlavaConfig.from_pretrained(MODEL_ZOOS[model_pth])
        with torch.device("cuda"):
            self.model = LlavaForCausalLM.from_pretrained(MODEL_ZOOS[model_pth], self.tokenizer, config=self.config)
        if "openai" in self.config.vision_encoder_name_or_path:
            image_processor_cls = CLIPImageProcessor
        elif "eva" in self.config.vision_encoder_name_or_path:
            image_processor_cls = EVACLIPImageProcessorWrapper
        self.image_processor = image_processor_cls.from_pretrained(
            MODEL_ZOOS[self.config.vision_encoder_name_or_path], local_files_only=True
        )

        self.conv_mode = "vicuna"

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "multi-choice":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        import copy

        from omni.constants import IGNORE_INDEX
        from omni.conversation.conv_templates import ConvTemplateManager
        from omni.conversation.conversation import Message, SeparatorStyle
        from omni.conversation.multimodal import MultimodalContent
        from omni.data.builders.builder_llava import process_images, truncate_and_replace
        from omni.models.llava.modeling_llava import KeywordsStoppingCriteria
        from omni.models.llava.tokenization_llava import DEFAULT_VISION_PLACEHOLDER_TOKEN

        conv = ConvTemplateManager[self.conv_mode]
        conv.append_message(Message(role=conv.roles[0], content=MultimodalContent(text="PLACEHOLDER")), verbose=False)
        conv.append_message(Message(role=conv.roles[1], content=None), verbose=False)
        prompt = conv.get_prompt()

        content, images = "", []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                content += DEFAULT_VISION_PLACEHOLDER_TOKEN + "\n"
                images.append(msg["value"])

        images = [Image.open(s).convert("RGB") for s in images]
        image_preprocessing = "pad"
        pixel_values = process_images(images, self.image_processor, image_preprocessing).to("cuda", dtype=self.model.dtype)

        prompt = prompt.replace("PLACEHOLDER", content)

        input_ids = self.tokenizer(prompt).input_ids
        labels = copy.deepcopy(input_ids)

        input_ids, labels = truncate_and_replace(
            input_ids=input_ids,
            labels=labels,
            replacement_dict={
                self.model.config.vision_placeholder_id: [self.model.config.vision_placeholder_id]
                * self.model.config.vision_token_len
            },
            labels_fill_value=IGNORE_INDEX,
            truncate=self.tokenizer.model_max_length,
        )
        input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.ADD_COLON_TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, pixel_values=pixel_values, stopping_criteria=[stopping_criteria], **self.kwargs
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
