# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import asyncio
import threading
from typing import List, Optional
import argparse
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tavily import TavilyClient
import time

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=16,
        cache_dir=f'cache/v1/hugggingface'
    )
    return corpus

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    if model_path in ['Qwen/Qwen3-Embedding-8B']:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = AutoModel.from_pretrained(model_path, attn_implementation="flash_attention_2",
                                          dtype=torch.float16).cuda()
    else:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        model.cuda()
        if use_fp16:
            model = model.half()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask=None,
        pooling_method="mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in
                              query_list]

        if 'qwen' in self.model_name.lower():
            if is_query:
                query_list = [f'Instruct: Given a search query, retrieve relevant passages\nQuery:{query}' for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        # inputs = {k: v.cuda() for k, v in inputs.items()}
        inputs.to(self.model.device)

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        elif 'qwen' in self.model_name.lower():
            output = self.model(**inputs)
            embeddings = last_token_pool(output.last_hidden_state, inputs['attention_mask'])
            query_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(None,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, eid: str = None):
        return self._batch_search(query_list, num, return_score, eid)


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        with open(config.example_id_file) as f:
            self.example_ids = json.load(f)
        # Thread lock for FAISS GPU operations (not thread-safe)
        self.faiss_lock = threading.Lock()

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        # FAISS GPU operations are not thread-safe, need to lock
        with self.faiss_lock:
            scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, eid: str = None):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            # FAISS GPU operations are not thread-safe, need to lock
            with self.faiss_lock:
                batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num: (i + 1) * num] for i in range(len(batch_idxs))]

            updated_batch_results = []
            updated_scores = []
            for one_batch_results,one_batch_scores in zip(batch_results,batch_scores):
                cur_batch_results = []
                cur_batch_scores = []
                for r,s in zip(one_batch_results,one_batch_scores):
                    # Filter by example_ids if eid is provided, otherwise include all results
                    if eid is None or int(r['id']) in self.example_ids.get(eid, []):
                        cur_batch_results.append(r)
                        cur_batch_scores.append(s)
                updated_batch_results.append(cur_batch_results)
                updated_scores.append(cur_batch_scores)

            results.extend(updated_batch_results)
            scores.extend(updated_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results,updated_batch_results,updated_scores
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
            self,
            retrieval_method: str = "bm25",
            retrieval_topk: int = 10,
            index_path: str = "./index/bm25",
            corpus_path: str = "./data/corpus.jsonl",
            dataset_path: str = "./data",
            data_split: str = "train",
            faiss_gpu: bool = True,
            retrieval_model_path: str = "./model",
            retrieval_pooling_method: str = "mean",
            retrieval_query_max_length: int = 256,
            retrieval_use_fp16: bool = False,
            retrieval_batch_size: int = 128,
            new_cache_dir: str = None,
            example_id_file: str = None,
            tavily_key: str = None
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.new_cache_dir = new_cache_dir
        self.example_id_file = example_id_file
        self.tavily_key = tavily_key


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    eid: str = None
    new_cache_dir: str = None

app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true,
      "eid": "example_id",
      "new_cache_dir": "cache/v1/hle"  # Optional: override cache directory per request
    }
    """
    try:
        assert len(request.queries)==1,"We now assume single query search"
        if not request.topk:
            request.topk = config.retrieval_topk  # fallback to default

        # Perform batch retrieval
        results, scores = retriever.batch_search(
            query_list=request.queries,
            num=1000,
            return_score=request.return_scores,
            eid=request.eid
        )
    except Exception as e:
        import traceback
        error_msg = f"Error in retrieve_endpoint: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        return {"error": str(e), "traceback": traceback.format_exc()}

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                if len(doc["content"])>100 and score>0.1:
                    combined.append({"document": doc, "score": score})
                if len(combined)>=request.topk:
                    break
            resp.append(combined)
        else:
            resp.append(single_result)
    if len(resp[0])<3:
        tavily_client = TavilyClient(config.tavily_key)
        
        # Use per-request cache directory if provided, otherwise use config default
        cache_base_dir = request.new_cache_dir if request.new_cache_dir else config.new_cache_dir
        
        # Check cache first (only if eid is provided)
        response = None
        search_idx = 0
        if request.eid:
            cache_dir = os.path.join(cache_base_dir, request.eid)
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # Try to load cached search results
            search_idx = 0
            cached_search_file = os.path.join(cache_dir, f"search_{search_idx}.json")
            if os.path.isfile(cached_search_file):
                try:
                    with open(cached_search_file, 'r') as f:
                        response = json.load(f)
                except Exception as cache_read_error:
                    # If cache read fails, will fetch fresh data below
                    pass
        
        # If no cached response, fetch from API
        if response is None:
            try:
                response = tavily_client.search(
                    query=request.queries[0],
                    search_depth="advanced",
                    max_results=20,
                    chunks_per_source=5
                )
            except Exception as tavily_search_error:
                return resp
            
            # Save search results to cache (only if eid is provided)
            if request.eid:
                # Find next available search index
                search_idx = 0
                while os.path.isfile(os.path.join(cache_base_dir,request.eid,f"search_{search_idx}.json")):
                    search_idx += 1
                # Save search results to cache
                with open(os.path.join(cache_base_dir,request.eid,f"search_{search_idx}.json"),'w') as f:
                    json.dump(response,f,indent=2)

        def extract_web(extract_argument):
            # Check cache first (only if eid is provided)
            extraction = None
            if request.eid:
                cached_extraction_file = os.path.join(cache_base_dir, request.eid, f"extraction_{search_idx}_{extract_argument['extract_id']}.json")
                if os.path.isfile(cached_extraction_file):
                    try:
                        with open(cached_extraction_file, 'r') as f:
                            extraction = json.load(f)
                    except Exception as cache_read_error:
                        # If cache read fails, will fetch fresh data below
                        pass
            
            # If no cached extraction, fetch from API
            if extraction is None:
                try:
                    extraction = tavily_client.extract(
                        urls=[extract_argument['url']],
                        extract_depth="advanced",
                        format="text"
                    )
                except Exception as tavily_extract_error:
                    return
                # Save extraction to cache (only if eid is provided)
                if request.eid:
                    with open(os.path.join(cache_base_dir,request.eid,f"extraction_{search_idx}_{extract_argument['extract_id']}.json"),'w') as f:
                        json.dump(extraction,f,indent=2)
            
            extract_argument['raw_extraction'] = extraction
            return extract_argument

        extraction_arguments = []
        # Only extract top results needed (limit to request.topk + buffer for filtering)
        # max_extractions = min(len(response['results']), max(request.topk * 2, 5))  # Extract 2x topk or at least 5, but not more than available
        # for extract_id,r in enumerate(response['results'][:max_extractions]):

        for extract_id,r in enumerate(response['results']):
            extraction_arguments.append([extract_web,{
                'extract_id': extract_id,
                'url': r['url'],
                'score': r['score']
            }])
        all_extraction_results = []
        for argument in extraction_arguments:
            all_extraction_results.append(argument[0](argument[1]))
        extraction_results = []
        for extraction_return in all_extraction_results:
            if not extraction_return:
                continue
            extract_content = ''
            for one_extraction_result in extraction_return['raw_extraction']["results"]:
                extract_content += one_extraction_result["raw_content"]+'\n\n'
            if len(extract_content.strip())>100:
                extraction_results.append([extract_content,extraction_return['score']])
            
        if len(extraction_results)>1:
            extraction_results = sorted(extraction_results,key=lambda x:x[1],reverse=True)
        # Only add top request.topk results to response
        for new_doc_id, new_search in enumerate(extraction_results[:request.topk]):
            assert isinstance(new_search,list)
            assert isinstance(new_search[0],str)
            resp[0].append({
                "document": {'content': new_search[0]},
                'score': -new_doc_id-1
            })
    return resp


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--new_cache_dir', type=str, default=f'cache/v1/hle')
parser.add_argument('--example_id_file', type=str, default='examples.json')
parser.add_argument('--tavily_key', type=str, default="")
parser.add_argument('--port', type=int)
args = parser.parse_args()

config = Config(
    retrieval_method='qwen',  # or "dense"
    index_path=os.path.join(os.environ.get('INDEX_DIR',None),'eval.index'),
    corpus_path=os.path.join(os.environ.get('INDEX_DIR',None),'eval.jsonl'),
    retrieval_topk=5,
    faiss_gpu=True,
    retrieval_model_path='Qwen/Qwen3-Embedding-8B',
    retrieval_pooling_method="mean",
    retrieval_query_max_length=32768,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
    new_cache_dir=args.new_cache_dir,
    example_id_file=args.example_id_file,
    tavily_key=args.tavily_key
)

retriever = DenseRetriever(config)

uvicorn.run(app, host="0.0.0.0", port=args.port)

