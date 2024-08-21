import os
import sys
import re
import sys
import json
import tempfile
import logging
import warnings

from typing import NoReturn, Iterable, Dict, Union, List, Generator, Optional

from datasets import Dataset
from transformers import (
    TrainingArguments,
    HfArgumentParser,
)

from datasets.utils.logging import disable_progress_bar as disable_datasets_pbar

from robust_deid.ner_datasets import DatasetCreator
from robust_deid.sequence_tagging import SequenceTagger
from robust_deid.sequence_tagging.arguments import (
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
)
from robust_deid.sequence_tagging.note_aggregate import NoteLevelAggregator
from deid_app.text_deid import TextDeid

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class JSONDatasetCreator(DatasetCreator):
    def __init__(
            self, 
            sentencizer: str, 
            tokenizer: str, 
            max_tokens: int = 128, 
            max_prev_sentence_token: int = 32, 
            max_next_sentence_token: int = 32, 
            default_chunk_size: int = 32, 
            ignore_label: str = 'NA'
        ) -> NoReturn:
        super().__init__(
            sentencizer, 
            tokenizer, 
            max_tokens, 
            max_prev_sentence_token, 
            max_next_sentence_token, 
            default_chunk_size, 
            ignore_label
        )

    def create_from_json_list(
        self,
        input_list: List,
        mode: str = 'predict',
        notation: str = 'BIO',
        token_text_key: str = 'text',
        metadata_key: str = 'meta',
        note_id_key: str = 'note_id',
        label_key: str = 'labels',
        span_text_key: str = 'spans'
    ) -> Iterable[Dict[str, Union[List[Dict[str, Union[str, int]]], List[str]]]]:
        """
        This function is used to get the sentences that will be part of the NER dataset.
        We check whether the note belongs to the desired dataset split. If it does,
        we fix any spans that can cause token-span alignment errors. Then we extract
        all the sentences in the notes, the tokens in each sentence. Finally we
        add some context tokens to the sentence if required. This function returns
        an iterable that iterated through each of the processed sentences
        Args:
            input_list (List): Input List of all texts. Make sure the spans from the items are in ascending order (based on start position)
            mode (str): Dataset being built for train or predict.
            notation (str): The NER labelling notation
            token_text_key (str): The key where the note text and token text is present in the json object
            metadata_key (str): The key where the note metadata is present in the json object
            note_id_key (str): The key where the note id is present in the json object
            label_key (str): The key where the token label will be stored in the json object
            span_text_key (str): The key where the note spans is present in the json object
        Returns:
            (Iterable[Dict[str, Union[List[Dict[str, Union[str, int]]], List[str]]]]): Iterate through the processed
                                                                                       sentences/training examples
        """
        # Go through the notes
        for note in input_list:
            note_text = note[token_text_key]
            note_id = note[metadata_key][note_id_key]
            
            # Skip to next note if empty string
            if not note_text:
                continue
            
            if mode == 'train':
                note_spans = note[span_text_key]
            # No spans in predict mode
            elif mode == 'predict':
                note_spans = None
            else:
                raise ValueError("Invalid mode - can only be train/predict")
            # Store the list of tokens in the sentence
            # Eventually this list will contain all the tokens in the note (split on the sentence level)
            # Store the start and end positions of the sentence in the note. This can
            # be used later to reconstruct the note from the sentences
            # we also store the note_id for each sentence so that we can map it back
            # to the note and therefore have all the sentences mapped back to the notes they belong to.
            sent_tokens = [sent_tok for sent_tok in self._dataset.get_tokens(
                text=note_text,
                spans=note_spans,
                notation=notation
            )]
            # The following loop goes through each sentence in the note and returns
            # the current sentence and previous and next chunks that will be used for context
            # The chunks will have a default label (e.g NA) to distinguish from the current sentence
            # and so that we can ignore these chunks when calculating loss and updating weights
            # during training
            for ner_sent_index, ner_sentence in self._sentence_dataset.get_sentences(
                    sent_tokens=sent_tokens,
                    token_text_key=token_text_key,
                    label_key=label_key
            ):
                # Return the processed sentence. This sentence will then be used
                # by the model
                current_sent_info = ner_sentence['current_sent_info']
                note_sent_info_store = {'start': current_sent_info[0]['start'],
                                        'end': current_sent_info[-1]['end'], 'note_id': note_id}
                ner_sentence['note_sent_info'] = note_sent_info_store
                yield ner_sentence

    

class GeneratorSequenceTagger(SequenceTagger):
    def __init__(
        self, 
        task_name, 
        notation, 
        ner_types, 
        model_name_or_path, 
        config_name: str | None = None, 
        tokenizer_name: str | None = None, 
        post_process: str = 'argmax', 
        cache_dir: str | None = None, 
        model_revision: str = 'main', 
        use_auth_token: bool = False, 
        threshold: float | None = None, 
        do_lower_case=False, 
        fp16: bool = False, 
        seed: int = 41, 
        local_rank: int = -1
    ):
        super().__init__(
            task_name, notation, ner_types, model_name_or_path, 
            config_name, tokenizer_name, post_process, cache_dir, 
            model_revision, use_auth_token, threshold, do_lower_case, 
            fp16, seed, local_rank
        )

    def set_predict_from_generator(
        self,
        dataset_generator: Generator,
        max_test_samples: Optional[int] = None,
        preprocessing_num_workers: Optional[int] = None,
        overwrite_cache: bool = False,
    ):
        itemslist = [item for item in dataset_generator]
        test_dataset = Dataset.from_list(itemslist)

        # Eval
        if max_test_samples is not None:
            test_dataset = test_dataset.select(range(max_test_samples))

        self._test_dataset = test_dataset.map(
            self._dataset_tokenizer.tokenize_and_align_labels,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )
    
    def predict(self, output_predictions_file: Optional[str] = None):
        # Predict without logging and storing the prediction
        # into file
        if self._test_dataset is not None and self._trainer is not None:
            self._logger.info("*** Predict ***")
            predictions, labels, metrics = self._trainer.predict(self._test_dataset, metric_key_prefix="predict")
            unique_note_ids = set()
            for note_sent_info in self._test_dataset['note_sent_info']:
                note_id = note_sent_info['note_id']
                unique_note_ids = unique_note_ids | {note_id}
            note_ids = list(unique_note_ids)
            note_level_aggregator = NoteLevelAggregator(
                note_ids=note_ids,
                note_sent_info=self._test_dataset['note_sent_info']
            )
            note_tokens = note_level_aggregator.get_aggregate_sequences(
                sequences=self._test_dataset['current_sent_info']
            )
            true_predictions, true_labels = self._post_processor.decode(predictions, labels)
            note_predictions = note_level_aggregator.get_aggregate_sequences(sequences=true_predictions)
            note_labels = note_level_aggregator.get_aggregate_sequences(sequences=true_labels)

            if output_predictions_file is None:
                return GeneratorSequenceTagger._SequenceTagger__get_predictions(
                    note_ids,
                    note_tokens,
                    note_labels,
                    note_predictions
                )
            else:
                GeneratorSequenceTagger._SequenceTagger__write_predictions(
                    output_predictions_file,
                    note_ids,
                    note_tokens,
                    note_labels,
                    note_predictions
                )
        else:
            if self._trainer is None:
                raise ValueError('Trainer not setup - Run setup_trainer')
            else:
                raise ValueError('Test data not setup - Run set_predict')


class ValuesTextDeid(TextDeid):
    def __init__(self, notation, span_constraint):
        super().__init__(notation, span_constraint)

    def run_deid_on_values(
            self,
            input_notes: List,
            predictions_list: List,
            deid_strategy,
            keep_age: bool = False,
            metadata_key: str = 'meta',
            note_id_key: str = 'note_id',
            tokens_key: str = 'tokens',
            predictions_key: str = 'predictions',
            text_key: str = 'text'
    ):
        # Store note_id to note mapping
        note_map = dict()
        for note in input_notes:
            note_id = note[metadata_key][note_id_key]
            note_map[note_id] = note
        # Go through note predictions and de identify the note accordingly
        for note in predictions_list:
            # Get the text using the note_id for this note from the note_map dict
            note_id = note[note_id_key]
            # Get the note from the note_map dict
            deid_note = note_map[note_id]
            # Get predictions
            predictions = self.decode(tokens=note[tokens_key], predictions=note[predictions_key])
            # Get entities and their positions
            entity_positions = self.get_predicted_entities_positions(
                tokens=note[tokens_key],
                predictions=predictions,
                suffix=False
            )
            yield ValuesTextDeid._TextDeid__get_deid_text(
                deid_note=deid_note,
                entity_positions=entity_positions,
                deid_strategy=deid_strategy,
                keep_age=keep_age,
                text_key=text_key
            )

class RobustDeID(object):
    
    def __init__(
        self,
        model,
        threshold,
        span_constraint='super_strict',
        sentencizer='en_core_sci_sm',
        tokenizer='clinical',
        max_tokens=128,
        max_prev_sentence_token=32,
        max_next_sentence_token=32,
        default_chunk_size=32,
        ignore_label='NA',
        disable_logs: bool = True,
    ):
        # disable all the loggings
        self.disable_logs = disable_logs
        if disable_logs:
            warnings.filterwarnings("ignore")

        # Create the dataset creator object
        self._dataset_creator = JSONDatasetCreator(
            sentencizer=sentencizer,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            max_prev_sentence_token=max_prev_sentence_token,
            max_next_sentence_token=max_next_sentence_token,
            default_chunk_size=default_chunk_size,
            ignore_label=ignore_label
        )

        parser = HfArgumentParser((
            ModelArguments,
            DataTrainingArguments,
            EvaluationArguments,
            TrainingArguments
        ))
        model_config = RobustDeID._get_model_config()
        model_config['model_name_or_path'] = RobustDeID._get_model_map()[model]
        if threshold == 'No threshold':
            model_config['post_process'] = 'argmax'
            model_config['threshold'] = None
        else:
            model_config['post_process'] = 'threshold_max'
            model_config['threshold'] = \
            RobustDeID._get_threshold_map()[model_config['model_name_or_path']][threshold]

        # sys.exit(0)
        # Ignore model config saving to file
        # with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        #     tmp.write(json.dumps(model_config) + '\n')
        #     tmp.seek(0)
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.

        # parse the arguments from the model config
        self._model_args, self._data_args, self._evaluation_args, self._training_args = parser.parse_dict(model_config)
        
        # Initialize the text deid object
        self._text_deid = ValuesTextDeid(notation=self._data_args.notation, span_constraint=span_constraint)
        # Initialize the sequence tagger
        self._sequence_tagger = GeneratorSequenceTagger(
            task_name=self._data_args.task_name,
            notation=self._data_args.notation,
            ner_types=self._data_args.ner_types,
            model_name_or_path=self._model_args.model_name_or_path,
            config_name=self._model_args.config_name,
            tokenizer_name=self._model_args.tokenizer_name,
            post_process=self._model_args.post_process,
            cache_dir=self._model_args.cache_dir,
            model_revision=self._model_args.model_revision,
            use_auth_token=self._model_args.use_auth_token,
            threshold=self._model_args.threshold,
            do_lower_case=self._data_args.do_lower_case,
            fp16=self._training_args.fp16,
            seed=self._training_args.seed,
            local_rank=self._training_args.local_rank
        )

        # Load the required functions of the sequence tagger
        self._sequence_tagger.load()
        self._disable_active_loggers()
    
    def _disable_active_loggers(self):
        loggers = [
            'robust_deid.sequence_tagging.sequence_tagger'
        ]

        if self.disable_logs:
            for lname in loggers:
                l = logging.getLogger(lname)
                l.setLevel(logging.ERROR)
            
            disable_datasets_pbar()
           
            
    def get_ner_dataset(self, notes_file):
        ner_notes = self._dataset_creator.create(
            input_file=notes_file,
            mode='predict',
            notation=self._data_args.notation,
            token_text_key='text',
            metadata_key='meta',
            note_id_key='note_id',
            label_key='label',
            span_text_key='spans'
        )
        return ner_notes

    def get_ner_dataset_from_json_list(self, notes_list: List):
        ner_notes = self._dataset_creator.create_from_json_list(
            input_list=notes_list,
            mode='predict',
            notation=self._data_args.notation,
            token_text_key='text',
            metadata_key='meta',
            note_id_key='note_id',
            label_key='label',
            span_text_key='spans'
        )
        return ner_notes
    
    def get_predictions(self, ner_notes_file):
        # Set the required data and predictions of the sequence tagger
        # Can also use self._data_args.test_file instead of ner_dataset_file (make sure it matches ner_dataset_file)
        self._sequence_tagger.set_predict(
            test_file=ner_notes_file,
            max_test_samples=self._data_args.max_predict_samples,
            preprocessing_num_workers=self._data_args.preprocessing_num_workers,
            overwrite_cache=self._data_args.overwrite_cache
        )
        # Initialize the huggingface trainer
        self._sequence_tagger.setup_trainer(training_args=self._training_args)
        # Store predictions in the specified file
        # if self.disable_logs:
        #     with HiddenPrints():
        #         predictions = self._sequence_tagger.predict()
        # else:
        predictions = self._sequence_tagger.predict()
        return predictions
    
    def get_predictions_from_generator(self, ner_notes_gen):
        # Set the required data and predictions of the sequence tagger
        # Can also use self._data_args.test_file instead of ner_dataset_file (make sure it matches ner_dataset_file)
        self._sequence_tagger.set_predict_from_generator(
            dataset_generator=ner_notes_gen,
            max_test_samples=self._data_args.max_predict_samples,
            preprocessing_num_workers=self._data_args.preprocessing_num_workers,
            overwrite_cache=self._data_args.overwrite_cache
        )
        # Initialize the huggingface trainer
        self._sequence_tagger.setup_trainer(training_args=self._training_args)

        # Store predictions in the specified file
        # if self.disable_logs:
        #     with HiddenPrints():
        #         predictions = self._sequence_tagger.predict()
        # else:
        predictions = self._sequence_tagger.predict()
        return predictions
    
    def get_deid_text_removed(self, notes_file, predictions_file):
        deid_notes = self._text_deid.run_deid(
            input_file=notes_file,
            predictions_file=predictions_file,
            deid_strategy='remove',
            keep_age=False,
            metadata_key='meta',
            note_id_key='note_id',
            tokens_key='tokens',
            predictions_key='predictions',
            text_key='text',
        )
        return deid_notes

    def get_deid_text_removed_from_values(self, notes_list, predictions_list):
        deid_notes = self._text_deid.run_deid_on_values(
            input_notes=notes_list,
            predictions_list=predictions_list,
            deid_strategy='remove',
            keep_age=False,
            metadata_key='meta',
            note_id_key='note_id',
            tokens_key='tokens',
            predictions_key='predictions',
            text_key='text',
        )
        return deid_notes
    
    def get_deid_text_replaced(self, notes_file, predictions_file):
        deid_notes = self._text_deid.run_deid(
            input_file=notes_file,
            predictions_file=predictions_file,
            deid_strategy='replace_informative',
            keep_age=False,
            metadata_key='meta',
            note_id_key='note_id',
            tokens_key='tokens',
            predictions_key='predictions',
            text_key='text',
        )
        return deid_notes

    def get_deid_text_replaced_from_values(self, notes_list, predictions_list):
        deid_notes = self._text_deid.run_deid_on_values(
            input_notes=notes_list,
            predictions_list=predictions_list,
            deid_strategy='replace_informative',
            keep_age=False,
            metadata_key='meta',
            note_id_key='note_id',
            tokens_key='tokens',
            predictions_key='predictions',
            text_key='text',
        )
        return deid_notes
    
    
    @staticmethod
    def _get_highlights(deid_text):
        pattern = re.compile('<<(PATIENT|STAFF|AGE|DATE|LOCATION|PHONE|ID|EMAIL|PATORG|HOSPITAL|OTHERPHI):(.)*?>>')
        tag_pattern = re.compile('<<(PATIENT|STAFF|AGE|DATE|LOCATION|PHONE|ID|EMAIL|PATORG|HOSPITAL|OTHERPHI):')
        text_list = []
        current_start = 0
        current_end = 0
        full_end = 0
        for match in re.finditer(pattern, deid_text):
            full_start, full_end = match.span()
            sub_text = deid_text[full_start:full_end]
            sub_match = re.search(tag_pattern, sub_text)
            sub_span = sub_match.span()
            tag_length = sub_match.span()[1] - sub_match.span()[0]
            yield (deid_text[current_start:full_start], None)
            yield (deid_text[full_start+sub_span[1]:full_end-2], sub_match.string[sub_span[0]+2:sub_span[1]-1])
            current_start = full_end
        yield (deid_text[full_end:], None)
    
    @staticmethod
    def _get_model_map():
        return {
            'OBI-RoBERTa De-ID':'obi/deid_roberta_i2b2',
            'OBI-ClinicalBERT De-ID':'obi/deid_bert_i2b2'
        }
    
    @staticmethod
    def _get_threshold_map():
        return {
            'obi/deid_bert_i2b2':{"99.5": 4.656325975101986e-06, "99.7":1.8982457699258832e-06},
            'obi/deid_roberta_i2b2':{"99.5": 2.4362972672812125e-05, "99.7":2.396420546444644e-06}
        }
    
    @staticmethod
    def _get_model_config():
        return {
            "post_process":None,
            "threshold": None,
            "model_name_or_path":None,
            "task_name":"ner",
            "notation":"BILOU",
            "ner_types":["PATIENT", "STAFF", "AGE", "DATE", "PHONE", "ID", "EMAIL", "PATORG", "LOC", "HOSP", "OTHERPHI"],
            "truncation":True,
            "max_length":512,
            "label_all_tokens":False,
            "return_entity_level_metrics":False,
            "text_column_name":"tokens",
            "label_column_name":"labels",
            "output_dir":"./run/models",
            "logging_dir":"./run/logs",
            "overwrite_output_dir":False,
            "do_train":False,
            "do_eval":False,
            "do_predict":True,
            "report_to":[],
            "per_device_train_batch_size":0,
            "per_device_eval_batch_size":16,
            "logging_steps":1000,
            "disable_tqdm":True,
            "log_level": 'error',
        }