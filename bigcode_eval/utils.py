import json
import math
import re
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any
from multiprocessing import Pool
from importlib.util import find_spec
# import anthropic

# import google.generativeai as genai

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

INFILL_MODE = False
INSTRUCTION_MODE = False


def is_openai_api(model: str) -> bool:
    return 'openai-' in model or 'gpt4-' in model or '-gpt4' in model or 'gpt-4' in model


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        limit_start=0,
        n_tasks=None,
        n_copies=1,
        prefix="",
        has_encoder=False,
        instruction_tokens=None,
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix
        self.has_encoder = has_encoder
        self.instruction_tokens = instruction_tokens

    def __iter__(self):
        prompts = []
        prompts_encoder = []
        infill = []
        instruction = []
        for sample in range(self.limit_start, self.limit_start + self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                infill.append(False)
                instruction.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    # Infilling mode
                    infill.append(True)
                    instruction.append(False)
                    prompt = self._make_infill_prompt(
                        **prompt_contents, preprefix=self.prefix
                    )
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    instruction.append(True)
                    infill.append(False)
                    prompt = self._make_instruction_prompt(
                        **prompt_contents, prefix=self.prefix
                    )
            else:
                 prompt = prompt_contents
            #    raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)
            if self.has_encoder:
                prompt_encoder = self.task.get_prompt_encoder(self.dataset[sample])
                if isinstance(prompt_encoder, str):
                    prompt_encoder = self.prefix + prompt_encoder
                prompts_encoder.append(prompt_encoder)

        if not len(set(infill)) == 1 or not len(set(instruction)) == 1:
            raise ValueError(
                "Mixed infill/instruction and completion prompts are not supported."
            )
        global INFILL_MODE
        global INSTRUCTION_MODE
        INFILL_MODE = infill[0]
        INSTRUCTION_MODE = instruction[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )
        if self.has_encoder:
            outputs_encoder = self.tokenizer(
                prompts_encoder,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
                return_token_type_ids=return_token_type_ids,
            )

        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                if self.has_encoder:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "ids_encoder": outputs_encoder.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                        "input_len_encoder": outputs_encoder.attention_mask[
                            sample
                        ].sum(),
                    }
                else:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                    }

    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")

    def _make_instruction_prompt(self, instruction, context, prefix=""):
        """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
        if not self.instruction_tokens:
            warnings.warn(
                "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
            )
            user_token, end_token, assistant_token = "", "", "\n"
        else:
            user_token, end_token, assistant_token = self.instruction_tokens
            if not user_token or not assistant_token or not end_token:
                warnings.warn(
                    "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
                )
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context
        )

        return prompt


def _parse_infill(code, tokenizer):
    """Reorder infill code and remove remaining special tokens."""
    model_id = tokenizer.name_or_path
    if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
        prefix, suffix, infill = code.split("<|mask:0|>", 2)
        infill = infill.split("<|endofmask|>")[0]
    elif model_id in ["bigcode/santacoder"]:
        prefix, rest = code.split("<fim-suffix>", 1)
        suffix, infill = rest.split("<fim-middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
        prefix, rest = code.split("<fim_suffix>", 1)
        suffix, infill = rest.split("<fim_middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    else:
        raise ValueError(f"Infilling not yet supported for: {model_id}")
    for k, v in tokenizer.special_tokens_map.items():
        if k == "additional_special_tokens":
            for t in v:
                infill = infill.replace(t, "")
        else:
            infill = infill.replace(v, "")
    return infill


def _parse_instruction(code, instruction_tokens):
    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]



class OpenAICompletionDataset(IterableDataset):
    """Preprocess the dataset for use with an OpenAI/VLLM completions endpoint
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: instruction-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        max_length: int,
        limit_start: int=0,
        n_tasks: Optional[int]=None,
        prefix: str="",
        instruction_tokens: Optional[Tuple[str, str, str]]=None,
    ):
        self.task = task
        self.dataset = dataset
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = n_tasks
        self.prefix = prefix
        self.instruction_tokens = instruction_tokens

    def __iter__(self):
        prompts = []
        for sample in range(self.limit_start, self.limit_start + self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    raise ValueError("Infilling mode is not supported in the OpenAI completions API")
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    prompt = self._make_instruction_prompt(
                        **prompt_contents, prefix=self.prefix
                    )
                else:
                    raise ValueError(f"Unsupported prompt keys: {prompt_contents.keys()}")
            else:
                prompt = prompt_contents
            #    raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)

        for sample in range(self.n_tasks):
            yield {
                "prompt": prompts[sample],
                "chat": self.dataset[sample]["chat"],
                "task_id": sample,
                "input_len": len(prompt),
            }

    def _make_instruction_prompt(self, instruction, context, prefix=""):
        """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
        if not self.instruction_tokens:
            warnings.warn(
                "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
            )
            user_token, end_token, assistant_token = "", "", "\n"
        else:
            user_token, end_token, assistant_token = self.instruction_tokens
            if not user_token or not assistant_token or not end_token:
                warnings.warn(
                    "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
                )
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context
        )
        return prompt



def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    limit_start=0,
    batch_size=20,
    prefix="",
    instruction_tokens=None,
    postprocess=True,
    is_wrapped=False,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """
    # keep track of the list of generated codes
    # where len(code_gens) = n_tasks and len(code_gens[0]) = number of generated code samples
    code_gens: List[List[Optional[str]]] = [[] for _ in range(n_tasks)]
    generations = [] if not intermediate_generations else intermediate_generations
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        with torch.no_grad():
            if task.stop_words:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                max_len = batch["input_len"].max().item()
                if "ids_encoder" in batch:
                    max_len += 1  # Add 1 for decoder_start_token_id
                gen_kwargs["stopping_criteria"][0].start_length = max_len
            if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
                idx = 1 if task.stop_words else 0
                gen_kwargs["stopping_criteria"][idx].input_length = (
                    batch["input_len"].max().item()
                )
            # import pdb; pdb.set_trace()
            inputs = batch["ids"][:, : batch["input_len"]] if tokenizer.padding_side == "right" else batch["ids"]
            if "ids_encoder" in batch:
                if is_wrapped:
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        decoder_input_ids=inputs,
                        input_ids=batch["ids_encoder"][:, : batch["input_len_encoder"]],
                        num_return_sequences=batch_size,
                        decoder_start_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **gen_kwargs,
                    )
                else:
                    # import pdb
                    # pdb.set_trace()
                    generated_tokens = model.generate(
                        decoder_input_ids=inputs,
                        input_ids=batch["ids_encoder"][:, : batch["input_len_encoder"]],
                        num_return_sequences=batch_size,
                        decoder_start_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **gen_kwargs,
                    )
            else:
                if is_wrapped:
                    # 8bit and 4bit models are wrapped in accelerator
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        input_ids=inputs,
                        num_return_sequences=batch_size,
                        **gen_kwargs,
                    )
                else:
                    generated_tokens = model.generate(
                        input_ids=inputs,
                        num_return_sequences=batch_size,
                        **gen_kwargs,
                    )
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_tokens)

            if save_every_k_tasks >= 1 and (step + 1) % save_every_k_tasks == 0:
                if not intermediate_save_generations_path:
                    raise ValueError(
                        "intermediate_save_generations_path cannot be empty!"
                    )

                code_gens = update_code_gens(
                    task,
                    tokenizer,
                    limit_start,
                    prefix,
                    instruction_tokens,
                    postprocess,
                    code_gens,
                    gen_token_dict,
                )
                with open(intermediate_save_generations_path, "w") as fp:
                    json.dump(generations + code_gens, fp)
                    print(
                        f"intermediate generations were saved at {intermediate_save_generations_path}"
                    )
                # reset gen_token_dict - prevent redundant decoding
                gen_token_dict = defaultdict(list)

    code_gens = update_code_gens(
        task,
        tokenizer,
        limit_start,
        prefix,
        instruction_tokens,
        postprocess,
        code_gens,
        gen_token_dict,
    )

    generations.extend(code_gens)
    return generations



DEFAULT_URL = (
    "https://corvoproxy.qa6.us-west-2.aws-dev.app.snowflake.com/v1/textcompletion"
)
import requests



def _corvo(args):

    (max_tokens, prompt, base_url, model, kwargs) = args
    org_prompt = prompt
    prompt = json.loads(prompt)
    system = json.dumps(prompt[0]['text'])
    user = json.dumps(prompt[1]['text'])

    if 'llama' in model:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n {system} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {user} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        prompt = f"[INST] {system} [/INST] {user}"

    params = {
        "prompts": [prompt],
        "maxOutputTokens": max_tokens,
        "temperature": 0.7,
        "modelDetails": {
            "name": model.split(':')[-1],
        },
        "stop": ['```\n'],
    }
    
    resp = requests.post(url=DEFAULT_URL, json=params)
    assert resp.status_code == 200
    data = resp.json()
    return (org_prompt, data["texts"][0][0])

def _process_completion(args):
    from lm_eval.models.openai_completions import oa_completion  # Lazy requirement
    import openai

    (max_tokens, prompt, base_url, model, kwargs) = args

    if 'corvo:' in model:
        return _corvo(args)

    if not find_spec("openai") or not find_spec("tiktoken"):
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
            "Please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`"
        )

    # Hack. All of our azure paths start with sfc.
    is_azure_openai = model.startswith("sfc")
    if is_azure_openai:
        client = openai.AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"], api_version=os.environ["OPENAI_VERSION"],
            azure_endpoint=os.environ["OPENAI_URL"])
        messages = [{"role": "user", "content": prompt}]
        try:
            completion = oa_completion(
                    client,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    chat=True,
                    **kwargs,
                )
            # oa_completion returns None in some failure cases.
            if completion:
                data = (
                    prompt,
                    completion.choices[0].message.content
                )
            else:
                data = (prompt, "openai completion is None")
        except openai.BadRequestError as e:
            error_str = f"Got error from openai: {e}"
            print(error_str)
            raise e
            # data = (prompt, error_str)
        return data
    else:
        try:
            messages = json.loads(prompt)
            def _map(r):
                if r == 'model':
                    return 'assistant'
                return r

            messages = [{"role": _map(p["role"]), "content": p["text"].strip()} for p in messages]

            if 'o1' in model:
                messages=[{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
        except:
            messages = [{"role": "user", "content": prompt}]
        

        
    
        if 'claude' in model:
            client = anthropic.Anthropic(api_key="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")
        elif 'gemini' in model or 'gemma' in model:
            genai.configure(api_key="AIzaSyB7n1aqiss47A1V-Sy_eP9kCt3ArAjFfxY")
            generation_config = {
                # "temperature": 1,
                # "top_p": 0.95,
                # "top_k": 40,
                "max_output_tokens": max_tokens,
                "response_mime_type": "text/plain",
            }

            client = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                # safety_settings = Adjust safety settings
                # See https://ai.google.dev/gemini-api/docs/safety-settings
                system_instruction=messages[0]["content"],
            )

            # 
        elif 'deepseek' in model:
            client = openai.OpenAI(api_key="sk-38700a6535144f449c4ca2414254c3a5", base_url="https://api.deepseek.com")
        else:
            client = openai.OpenAI(base_url=base_url)

        if 'temperature' in kwargs:
            kwargs['temperature'] = float(kwargs['temperature'])

        if 'top_p' in kwargs:
            kwargs['top_p'] = float(kwargs['top_p'])
        
        if 'o1' not in model:
            kwargs['stop'] = ['```\n']

        
        if 'claude' in model:
            completion = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=messages[0]['content'],
                messages=messages[1:],
                stop_sequences=['```\n'],
            )
            data = (prompt, completion.content[0].text)
        elif 'gemini' in model or 'gemma' in model:
            chat = client.start_chat()

            try:
                completion = chat.send_message(messages[1]["content"])
                data = (prompt, completion.text)
            except genai.types.generation_types.StopCandidateException:
                print('Recitation error')
                data = (prompt, '')
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs,
            )

            if completion:
                data = (
                    prompt,
                    completion.choices[0].message.content
                )
            else:
                data = (prompt, "completion is None")

        return data


def complete_code_api(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    limit_start=0,
    batch_size=20,
    prefix="",
    instruction_tokens=None,
    postprocess=True,
    is_wrapped=False,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
    args=None,
    **gen_kwargs,
) -> Tuple[List[Dict[str, Any]], List[str]]:

    code_gens: List[List[Optional[str]]] = [[] for _ in range(n_tasks)]
    generations = [] if not intermediate_generations else intermediate_generations
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens

    reqs = []
    for step, batch in enumerate(dataloader):
        prompt = batch["prompt"][0]
        reqs.append(prompt)
    if False: # isinstance(task, CortexAnalystEval):
        reqs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": req}],
                add_generation_prompt=True,
                tokenize=False
            )
            for req in reqs
        ]

    code_gens: List[List[Optional[str]]] = [[] for _ in range(n_tasks)]
    generations = [] if not intermediate_generations else intermediate_generations

    responses = []
    indices = []

    kwargs = {}
    if "temperature" in args.model_args:
        kwargs["temperature"] = args.model_args["temperature"]
    if "top_p" in args.model_args:
        kwargs["top_p"] = args.model_args["top_p"]

    _TOKEN_COUNT_LEEWAY = 4

    def _compute_max_tokens(prompt: str) -> int:
        if args.max_new_tokens is not None:
            # If both are specified, `max_new_tokens` takes precedence.
            return args.max_new_tokens
        elif args.max_length_generation is not None:
            prompt_token_count = len(tokenizer.tokenize(prompt))
            if prompt_token_count >= args.max_length_generation:
                raise ValueError(
                    f"The prompt ({prompt_token_count} tokens) is longer than the total budgeted "
                    f"max_length_generation ({args.max_length_generation} tokens)"
                )
            return args.max_length_generation - prompt_token_count - _TOKEN_COUNT_LEEWAY
        return None

    with Pool(batch_size) as p:
        run_for = [
                    (_compute_max_tokens(prompt), prompt, args.model_args['base_url'] if 'base_url' in args.model_args else None, args.model_args['model'], kwargs)
                    for prompt in reqs
                ]

        # _process_completion(run_for[0])
        # import pdb
        # pdb.set_trace()
        for step, (prompt, response) in enumerate(tqdm(
            p.imap(
                _process_completion,
                run_for
            ),
            total=n_tasks,
        )):
            # For cortex analyst, it is much easier to post process if we only include the response and not the prompt.
            if False: # isinstance(task, CortexAnalystEval):
                responses.append(response)
            else:
                responses.append(prompt+response)
            indices.append(step)

            generated_tokens = tokenizer.encode(responses[step])  # For compatibility with postprocessing
            gen_token_dict[step].append(generated_tokens)

            if save_every_k_tasks >= 1 and (step + 1) % save_every_k_tasks == 0:
                if not intermediate_save_generations_path:
                    raise ValueError(
                        "intermediate_save_generations_path cannot be empty!"
                    )

                code_gens = update_code_gens(
                    task,
                    tokenizer,
                    limit_start,
                    prefix,
                    instruction_tokens,
                    postprocess,
                    code_gens,
                    gen_token_dict,
                )

                with open(intermediate_save_generations_path, "w") as fp:
                    json.dump(generations + code_gens, fp)
                    print(
                        f"intermediate generations were saved at {intermediate_save_generations_path}"
                    )
                # reset gen_token_dict - prevent redundant decoding
                gen_token_dict = defaultdict(list)

    code_gens = update_code_gens(
        task,
        tokenizer,
        limit_start,
        prefix,
        instruction_tokens,
        postprocess,
        code_gens,
        gen_token_dict,
    )

    generations.extend(code_gens)

    records = None # get_prompt_completion_records(reqs, generations)
    return generations


def update_code_gens(
    task,
    tokenizer,
    limit_start,
    prefix,
    instruction_tokens,
    postprocess,
    code_gens,
    gen_token_dict,
):  
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            if INFILL_MODE or tokenizer.eos_token in task.stop_words:
                if s[0] == tokenizer.bos_token_id:
                    s = s[1:]
                # Treat eos token as a regular stop word not removing it from the output
                # If it's removed it may have the effect of removing it in the middle of a
                # longer generation in case a batch size > 1 is used, which will result in
                # a wrong generation as it won't be used for splitting lateron
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                try:
                    # some tokenizers add a multi-token prefix to the generation (e.g ChatGLM)
                    tokenizer_prefix = tokenizer.decode(tokenizer.get_prefix_tokens())
                    if gen_code.startswith(f"{tokenizer_prefix}"):
                        gen_code = gen_code[len(tokenizer_prefix):].lstrip()
                except:
                    pass
                if INFILL_MODE:
                    gen_code = _parse_infill(gen_code, tokenizer)
                if INSTRUCTION_MODE:
                    gen_code = _parse_instruction(gen_code, instruction_tokens)
            else:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            if not INFILL_MODE:
                gen_code = gen_code[len(prefix) :]
            if postprocess:
                code_gens[sample].append(
                    task.postprocess_generation(gen_code, int(sample) + limit_start)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[sample].append(gen_code)
    return code_gens


def remove_after_return(code):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0: start_match]
        end_last_match = end_match
    return code
