"""Desc
"""
import fcntl
import json
import logging
from pathlib import Path
import pandas as pd
import pickle
import re

from bigcode_eval.tasks.custom_metrics.ds_f_eng_eval import tabpfn_average_accuracy
from bigcode_eval.tasks.custom_metrics.ds_f_eng_eval_3_refactored import DataProcessor, \
    XGBoostModel, TabPFNModel, Evaluator, ScoreNormalizer
from bigcode_eval.base import Task

_CITATION = """"""


class DsFEng(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "/home/yak/bigcode-evaluation-harness/dataset.jsonl"
    DATAFRAME_PATH = "/home/yak/bigcode-evaluation-harness/final_11"
    METADATA_PATH = "/home/yak/bigcode-evaluation-harness/meta11.csv"
    LOGGING_PATH = "/home/yak/bigcode-evaluation-harness/final_9/log.jsonl"
    BASELINE_SCORES = "/home/yak/bigcode-evaluation-harness/final_9/baselines_scores9.pickle"
    DATAFRAMES_URL = ""

    def __init__(self, timeout: float = 3.0):
        super().__init__(
            stop_words=["\n```"],  # "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"],
            requires_execution=True,
        )
        self._ds_f_eng_dir = Path(__file__).parent / "ds_f_eng"
        self._ds_f_eng_dir.mkdir(parents=True, exist_ok=True)
        self._dataframes_dir = Path(self.DATAFRAME_PATH)  # self._ds_f_eng_dir / "ds_f_eng_dataframes"
        self.metadata = pd.read_csv(self.METADATA_PATH)
        self.timeout = timeout
        # Loading the defaultdict from a file
        try:
            with open(self.BASELINE_SCORES, 'rb') as handle:
                self.baseline_scores = pickle.load(handle)
        except FileNotFoundError:
            self.baseline_scores = {}
            print(f"file {self.BASELINE_SCORES} not found. Will save there current baseline outputs.")
        self.score_normalizer = ScoreNormalizer(self.baseline_scores)
        # TODO(ajedrosz): fix once move to hub
        # self._download_dataframes()

    def _download_dataframes(self):
        lock = self._dataframes_dir.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._ds_f_eng_dir.exists():
                logging.info(f"Downloading DS-F-ENG dataframes to {self._ds_f_eng_dir}")
                request = requests.get(DsFEng.DATAFRAMES_URL, stream=True)
                zip_file = zipfile.ZipFile(io.BytesIO(request.content))
                zip_file.extractall(self._ds_f_eng_dir)
                logging.info(f"Successfuly extracted all DS-F-ENG dataframes to {self._dataframes_dir}")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # TODO(ajedrosz): fix once move to hub
        return self.dataset["train"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        # return doc["chat"]
        return "\n\n".join([i["text"] for i in doc["chat"]]) # + '\n\n```python\n' # doc["prompt"]

    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        return doc["dataframe_id"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        # import pdb
        # pdb.set_trace()
        code = re.findall(
            r"```python\n([^`]+)(```)?", generation, re.MULTILINE | re.DOTALL
        )[-1][0]
        # dataset = self.get_dataset()
        # prompt = self.get_prompt(dataset[idx])
        # generation = generation[len(prompt):]
        return code
        # return self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        references = [r.replace("/", "__") + ".csv" for r in references]
        data_processor = DataProcessor(Path(self._dataframes_dir), timeout=60, allow_no_transform=True)

        evaluator = Evaluator(data_processor, self.LOGGING_PATH, self.metadata)

        results = evaluator.evaluate(generations, references, do_baseline=False)
        print(results)

        normalized_score = self.score_normalizer(results)
        print(f"ds_f_eng normalized_score: {normalized_score}")
        return normalized_score
