import numpy as np
import torch
import torch.nn as nn

from nltk import sent_tokenize
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any

from example.utterance import ExampleUtterance
from templates.actor import Actor
from templates.metrics import BaseEvalMetric


# Toggle to decide whether to use GPU in inference
USE_CUDA = True
CUDA = USE_CUDA and torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'

# Configuration for tokenization
ENCODE_CONFIG = {
    "return_tensors": "pt",
    "max_length": 1024,
    "truncation": True,
    "padding": True,
}

class UniEvalDialogMetric(BaseEvalMetric):
    def __init__(self, model_type, context_len=1) -> None:
        super().__init__()
        print(f"Initialising UniEvalDialogMetric")

        self.scorer = UniEvaluator(model_type)
        self.dimensions = ['naturalness', 'coherence', 'engagingness', 
                           'groundedness', 'understandability']
        self.context_len = context_len
        
        print(f"Finished loading UniEvalDialogMetric")

    def score_utterance(self, utterance: ExampleUtterance, context: List[ExampleUtterance], **kwargs) -> Dict[str, Any]:
        # Scores a single turn of the conversation
        eval_scores = {}
        eval_dims = self.dimensions
        
        for dim in eval_dims:
            # Calculate summation score for 'engagingness'
            if dim == 'engagingness':
                output_list, context_list = [], []
                n_sents = [] # the number of sentences in each generated response
                src_list = [x.text for x in context[-self.context_len:]]
                context = kwargs['additional_context_unieval'] if kwargs else '' # empty-string if no additional input is given
                system_outputs = sent_tokenize(utterance.text)
                n_sents.append(len(system_outputs))
                for i in range(len(system_outputs)):
                        context_list.append(context)
                        output_list.append(system_outputs[i])

                input_list = add_question(dimension=dim, output=output_list, 
                                          src=src_list, context=context_list)
                sent_score = self.scorer.score(input_list)
                
                # Get the summation score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]))
                    start_idx += cur_n_sent
                
            
            # Calculate turn-level score for other dimensions
            elif dim in ['naturalness', 'coherence', 'groundedness', 'understandability']:
                src_list = [x.text for x in context[-self.context_len:]]
                output_list = [utterance.text]
                context_list = [kwargs['additional_context_unieval']] if kwargs else ['']
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, context=context_list)
                score = self.scorer.score(input_list)
        
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')
            
            eval_scores[dim] = score[0]

        eval_scores['overall'] = np.mean(list(eval_scores.values()))
        return eval_scores

    def score_conversation(self, conversation: List[ExampleUtterance],
                           store: Dict[str, Any] = None, **kwargs) -> float:
        # Scores bot responses across entire conversation, returns average turn-wise scores
        dims = self.dimensions + ['overall']
        scores = dict(zip(dims, [0]*len(dims)))
        overall_scores_per_turn = []    
        turns = 0

        for i, utterance in enumerate(conversation):
            if utterance.actor == Actor.BOT: # only evaluate bot responses
                turns += 1
                turn_score = self.score_utterance(utterance=utterance, context=conversation[:i])
                overall_scores_per_turn.append(turn_score['overall'])
                scores = {x: scores.get(x, 0) + turn_score.get(x, 0) for x in set(scores).union(turn_score)}

        # Enable return of scores per turn on top of document level score using store object
        if store is not None and "scores_per_turn" in store:
            store['scores_per_turn'] = overall_scores_per_turn

        conversation_scores = {k: v / turns for k, v in scores.items()}
        return conversation_scores['overall']


class UniEvaluator:
    def __init__(self, model_type) -> None:
        print(f"Loading UniEval Model")
        self.device = DEVICE

        self.config = AutoConfig.from_pretrained(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_type, config=self.config)

        self.model.eval()
        self.model.to(self.device)

        self.softmax = nn.Softmax(dim=1)

        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]


    def score(self, inputs: List[Any], batch_size: int = 8) -> List[Any]:
        # Get scores for the given samples.
        # final_score = postive_score / (postive_score + negative_score)

        # The implementation of "forward" in T5 still requires decoder_input_ids.
        # Therefore, we construct a random one-word target sequence.
        # The content of the target has no effect on the final scores.
        tgts = ["No" for _ in range(len(inputs))]

        pos_score_list, neg_score_list = [], []
        for i in range(0, len(inputs), batch_size):
            src_list = inputs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        **ENCODE_CONFIG,
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        **ENCODE_CONFIG,
                    )

                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)[:, 0].unsqueeze(-1)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels = tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
            
                    pos_score = self.softmax(logits)[:, self.pos_id] # Yes
                    neg_score = self.softmax(logits)[:, self.neg_id] # No

                    cur_pos_score = [x.item() for x in pos_score]
                    cur_neg_score = [x.item() for x in neg_score]
                    pos_score_list += cur_pos_score
                    neg_score_list += cur_neg_score

            except RuntimeError:
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        
        score_list = []
        for i in range(len(pos_score_list)):
            score_list.append(pos_score_list[i] / (pos_score_list[i] + neg_score_list[i]))
            
        return score_list


def add_question(dimension: str, output: List[str], src: List[Any] = None, context: List[str] = None) -> List[str]:
    # Adds questions to generate input in Bool-QA format for UniEval.
    input_with_question = []
    if not src:
        src = ['']
    for i in range(len(output)):
        if dimension == 'naturalness':
            cur_input = 'question: Is this a natural response in the dialogue? </s> response: ' + output[i]
        elif dimension == 'coherence':
            cur_input = 'question: Is this a coherent response given the dialogue history? </s> response: '\
                        + output[i] + ' </s> dialogue history: ' + src[i]
        elif dimension == 'engagingness':
            cur_input = 'question: Is this an engaging and informative response according to the dialogue history and fact? </s> response: '\
                        + output[i] + ' </s> dialogue history: ' + src[i] + ' </s> fact: ' + context[i]
        elif dimension == 'groundedness':
            cur_input = 'question: Is this response consistent with knowledge in the fact? </s> response: '\
                            + output[i] + ' </s> fact: ' + context[i]
        elif dimension == 'understandability':
            cur_input = 'question: Is this an understandable response in the dialogue? </s> response: ' + output[i]
        else:
            raise NotImplementedError('The input format for this dimension is still undefined. Please customize it first.')

        input_with_question.append(cur_input)

    return input_with_question