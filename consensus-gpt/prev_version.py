#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:59:35 2023

@author: sidvijay
"""

import json
import openai
import os
import pinecone
import datetime
import boto3
from boto3.dynamodb.conditions import Key
import tiktoken
from tqdm.auto import tqdm
import time
import uuid
import anthropic
from langchain.embeddings.openai import OpenAIEmbeddings
import xml.etree.ElementTree as ET
import random
import re

client = boto3.client('kendra')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

questions_table = dynamodb.Table('6998-questions-db')
comments_table = dynamodb.Table('6998-comments-db')
cache_table = dynamodb.Table('consensus-cache')
search_table = dynamodb.Table('consensus-searches')


def lambda_handler(event, context):
    #print(event)
    #query = json.loads(event['body'])["query"]
    #sessionID = json.loads(event['body'])["sessionID"]
    
    query = event["queryStringParameters"]["q"]
    sessionID = "1232321"
    
    rand = random.randint(0, 1)
    
    if rand % 2 == 0:
        openai.api_key = "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"
        os.environ["OPENAI_API_KEY"] = "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"
    else:
        openai.api_key = "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"
        os.environ["OPENAI_API_KEY"] = "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"
    
    
    
    #TIME
    start_time = time.time()

    PINECONE_CACHE_KEY='5d2e9d8d-af18-465e-a3de-1a25c0a8ea93'
    PINECONE_CACHE_ENV='us-west1-gcp-free'

    pinecone.init(
        api_key=PINECONE_CACHE_KEY,
        environment=PINECONE_CACHE_ENV
    )


    index_cache = pinecone.Index("index-cache")


    questions = []
    PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
    PINECONE_ENV = 'us-central1-gcp'


    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)
    xq = openai_embed.embed_query(query)
    

    xc = index_cache.query(xq, top_k=10, include_metadata=True)
    written_dynamo_db = False
    try: 
        for result in xc['matches']:
            if result['score'] < .98:
                print('THIS IS RUNNING')
                str_id = f'{uuid.uuid4()}'
                record = (str_id, xq, {'text': query})
                index_cache.upsert(vectors=[record])
                written_dynamo_db = True
                break

            response = cache_table.query(
                IndexName='question-index',
                KeyConditionExpression=Key('question').eq(result['metadata']['text'])
            )

            #print(result['metadata']['text'])
            answer = response['Items'][0]['response']

            print("CACHE USED BYEEEE")

            #row['sessionID'] = sessionID
            #row['query'] = query
            with search_table.batch_writer() as batch:
                    print('WRITING TO DYNAMODB')
            #        batch.put_item(Item=row)

            end_time = time.time()
            time_diff = end_time - start_time
            print("GET CACHE PINECONE: {:.3f} seconds".format(time_diff))
    
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,GET',
                },
                'body':  json.dumps(answer)
            }
        if len(xc['matches']) == 0:
            str_id = f'{uuid.uuid4()}'
            record = (str_id, xq, {'text': query})
            index_cache.upsert(vectors=[record])
            written_dynamo_db = True

            print("UPDATED PINECONE")

    except:
        print('ERROR SINCE NO INDEX')
        str_id = f'{uuid.uuid4()}'
        record = (str_id, xq, {'text': query})
        index_cache.upsert(vectors=[record])
        written_dynamo_db = True

        
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET CACHE PINECONE: {:.3f} seconds".format(time_diff))
    
    # query dynamo db to get all comments
    start_time = time.time()

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    index = pinecone.Index("semantic-search-6998")
    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)


    # query Pinecone to get all Questions
    questions = []
    xc = index.query(xq, top_k=10, include_metadata=True)
    total_tokens = 0

    for result in xc['matches']:
        if result['score'] < .8:
            break

        response = questions_table.query(
            IndexName='title-index',
            KeyConditionExpression=Key('title').eq(result['metadata']['text'])
        )

        #print(result['metadata']['text'])
        for item in response['Items']:
            questions.append(item['id'])

    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET QUESTIONS PINECONE: {:.3f} seconds".format(time_diff))
    
    # query dynamo db to get all comments
    start_time = time.time()

    comments = []
    totalTokensOver = False

    for result in questions:
        response = comments_table.query(
            IndexName='parent_id-index',
            KeyConditionExpression=Key('parent_id').eq(result)
        )
        for item in response['Items']:
            num_tokens = num_tokens_from_string(item['body'],"cl100k_base")
            total_tokens +=  num_tokens

            if total_tokens > 3600:
                totalTokensOver = True
                break

            comments.append(item['body'])
            
        if totalTokensOver:
            break

    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET COMMENTS DYNAMODB: {:.3f} seconds".format(time_diff))    

    # GPT KEY CLAIMS, SUMMARY, SUPPORTING COMMENTS -------------------------------------------------------------------------
    start_time = time.time()
    
    
    comments_str = "\n".join([f"<comment id='{i+1}'>{comment}</comment>" for i, comment in enumerate(comments)])
    
    
    GPT_prompt = f''' 
    Here's a series of numbered Reddit comments in response to the question "{query}" by a user: 
    <comments>
    {comments_str}
    </comments>
    
    You are an AI language model tasked with analyzing the provided numbered Reddit comments in response to the question "{query}". 
    You must perform the following task:
    
    1. Generate an in-depth, comprehensive answer to the question using the various opinions expressed in the comments. 
    This answer should capture the spectrum of viewpoints, highlighting key arguments. 
    It should also provide explanations for complex or contentious points and, when necessary, illustrate these points with examples.
    Do not specifically cite specific comments. 
    Seperate your answer into logical sections where necessary. 
    At the end, include two sections called "areas of consensus" and "areas of disagreement". 
    Your answer must be at most 150 words.
    
    '''
    
    GPT_prompt = f''' 
    You are an AI language model examining a set of numbered Reddit comments in response to the user question: "{query}". 
    <comments>
    {comments_str}
    </comments>
    
    Your task involves the following:
    
    Construct a comprehensive, in-depth answer to the question utilizing the array of opinions presented in the comments. 
    Your answer should encapsulate the range of perspectives and emphasize key arguments, 
    clarifying complex or disputed points with examples, as needed. Avoid direct citation of specific comments.
    Organize your answer into intuitive sections as necessary, enhancing readability and understanding. 
    Ensure to include two distinct sections named "areas of consensus" and "areas of disagreement" at the end.
    
    '''
    
    GPT_prompt = f''' 
    You are an AI language model examining a set of numbered Reddit comments in response to the user question: "{query}". 
    <comments>
    {comments_str}
    </comments>
    
    Your task involves the following:
    
    Construct a comprehensive, in-depth answer to the question utilizing the range of perspectives and arguments presented in the comments. 
    Organize your answer into intuitive sections as necessary, enhancing readability and understanding. 
    
    '''
    
    
    GPT_prompt = f''' 
    You are an AI language model examining a set of numbered Reddit comments in response to the user question: "{query}". 
    <comments>
    {comments_str}
    </comments>
    
    You must perform the following tasks:
    
    Construct a comprehensive, in-depth answer to the question utilizing the range of perspectives and arguments presented in the comments. 
    Organize your answer into intuitive sections as necessary, enhancing readability and understanding. 
    
    '''
    
    
    
    
    
    
    
    
    GPT_prompt4 = f"""
    Here's a series of numbered Reddit comments in response to the question "{query}" by a user: 
    <comments>
    {comments_str}
    </comments>
    
    You are an AI language model tasked with analyzing the provided numbered Reddit comments in response to the question "{query}". You must perform the following tasks:
        
    Provide a thorough, in-depth response to the question, capturing the wide range of perspectives expressed in the comments. Structure your answer into sections based on distinct key arguments made. For each section, include:
       1. A succinct claim that encapsulates the key argument responding to the question.
       2. An extensive analysis of the key argument, elucidating any intricate or disputed aspects, and drawing on comment examples when needed to illuminate these points.
       3. A ranked list of comment IDs that present the argument, ordered by their relevance to the claim.
    
    
    Your output should be an XML block as follows:
    
    <llm_resp_array>
        <sections>
            <section>
                <section_number>1</section_number>
                <claim>Key claim for section 1</claim>
                <explanation>Explanation for section 1</explanation>
                <comments_in_cluster>Input comment IDs for section 1 here (list)</comments_in_cluster>
            </section>
            <section>
                <section_number>2</section_number>
                <claim>Key claim for section 2</claim>
                <explanation>Explanation for section 2</explanation>
                <comments_in_cluster>Input comment IDs for section 2 here (list)</comments_in_cluster>
            </section>
            <!-- Add further sections and their respective details as necessary -->
        </sections>
    </llm_resp_array>
    
    You must only output this XML block. Nothing else.
    
    """
    
    
    
    GPT_prompt3 = f'''
    
    Here's a series of numbered Reddit comments in response to the question "{query}" by a user: 
    <comments>
    {comments_str}
    </comments>
    
    You are an AI language model tasked with analyzing the provided numbered Reddit comments in response to the question "{query}". You must perform the following task:
    
    1. Generate an in-depth, comprehensive answer to the question using the various opinions expressed in the comments. This summary should capture the full spectrum of viewpoints, highlighting key arguments, and identifying areas of consensus or disagreement. It should also provide explanations for complex or contentious points and, when necessary, illustrate these points with examples from the comments. Seperate your answer into logical sections where necessary. 
        
    '''
    
    
    
    GPT_prompt = f''' Here's a series of numbered Reddit comments in response to the question "{query}" by a user: 
    <comments>
    {comments_str}
    </comments>
    
    You are an AI language model tasked with analyzing the provided numbered Reddit comments in response to the question "{query}". You must perform the following tasks:
    
    1. Construct a comprehensive, in-depth answer to the question utilizing the range of perspectives and arguments presented in the comments. Organize your answer into intuitive sections as necessary, enhancing readability and understanding. 
    
    2. Create five distinct groups, or "clusters", out of the comments. 
    Each cluster should represent a unique key claim or central argument that is common to the comments within that cluster. 
    Each claim must answer the question "{query}". You must form at least 5 different clusters. 
    Each cluster should be made up of multiple comments that share the same key claim. 
    If a cluster contains only a single comment, do not include it in the final output. 
    For each identified cluster, provide the overarching key claim that unites the comments in this cluster, a detailed summarization that explains the main points made in the comments of this cluster, and the comment id of the comments that belong to this cluster (ordered by their relevance to the claim) outputted as a list.
    
    Your response for task 1 and task 2 should be independent of each other. 
    Exclude any troll-like or irrelevant remarks, focusing primarily on thoughtful, substantial responses to the question. The final output should be formatted into a neatly structured XML block, capturing your comprehensive analysis.
    
    The desired response format should look like:
    
    <llm_resp_array>
    <summary_and_analysis>
        <answer>
        <!-- Your comprehensive, in-depth answer goes here -->
        </answer>
        <clusters>
            <cluster>
                <cluster_number>1</cluster_number>
                <claim>Shared claim in cluster 1</claim>
                <summary>Your cluster 1 summary</summary>
                <comments_in_cluster>List of comment IDs</comments_in_cluster>
            </cluster>
            <cluster>
                <cluster_number>2</cluster_number>
                <claim>Shared claim in cluster 2</claim>
                <summary>Your cluster 2 summary</summary>
                <comments_in_cluster>List of comment IDs</comments_in_cluster>
            </cluster>
            <!-- Additional clusters and their details go here -->
        </clusters>
    </summary_and_analysis>
    </llm_resp_array>
    
    You must only output this XML block. Nothing else.
    
    '''
    
    
    
    
    
    
    
    
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{GPT_prompt}"},
        ],
      temperature=0.3,
    )
    
    # Extract the response content
    output = response['choices'][0]['message']['content']
    print(output)
    print(len(comments))
    
    # Parse XML output
    root = ET.fromstring(output)
    
    # Initialize a dictionary for JSON output
    json_output = {"llm_resp_array": {"summary_and_analysis": {"clusters": []}}}
    
    # Extract data and replace IDs with their respective comments
    for cluster in root.find('summary_and_analysis').find('clusters'):
        cluster_number = cluster.find('cluster_number').text
        claim = cluster.find('claim').text
        summary = cluster.find('summary').text

        comment_ids = re.sub(r"[^\d,]", "", cluster.find('comments_in_cluster').text).split(',')
        comment_ids = [id.strip() for id in comment_ids if id.strip()]
        comment_texts = [comments[int(id)-1] for id in comment_ids]
    
        # Append the cluster to the JSON output
        json_output["llm_resp_array"]["summary_and_analysis"]["clusters"].append({
            "cluster_number": cluster_number,
            "claim": claim,
            "summary": summary,
            "comments_in_cluster": "; ".join(comment_texts)
        })
    
    # Convert the dictionary to a JSON string
    json_output = json.dumps(json_output, indent=4)
    #print(json_output)
    
    end_time = time.time()
    time_diff = end_time - start_time

    print("LLM: {:.3f} seconds".format(time_diff))

    input_params = {
        "comments": comments,
        "llm_resp_array" : json_output
    }
    #print(input_params)
    
    

    # write to pinecone cache in dynamoDB if not exists
    row = {}
    row['id'] = uuid.uuid1().hex
    row['question'] = query
    row['response'] = response

    if written_dynamo_db:
        with cache_table.batch_writer() as batch:
            print('WRITING TO DYNAMODB')
            batch.put_item(Item=row)



    row['sessionID'] = sessionID
    with search_table.batch_writer() as batch:
            print('WRITING TO DYNAMODB')
            batch.put_item(Item=row)


    # return search results and headers
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,GET',
        },
        'body':  json.dumps(response)
    }


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens