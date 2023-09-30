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
from datetime import datetime
import random
import re
import pandas as pd
 

client = boto3.client('kendra')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

questions_table = dynamodb.Table('lynk-questions')
comments_table = dynamodb.Table('lynk-comments')
cache_table = dynamodb.Table('lynk-summary-cache')
search_table = dynamodb.Table('consensus-searches')
topics_table = dynamodb.Table('consensus-topics')
user_search_table = dynamodb.Table('consensus-searches-daily')



def lambda_handler(event, context):
    print(event)
    #query = json.loads(event['body'])["query"]
    # sessionID = json.loads(event['body'])["sessionID"]
    query = event["query"]
    sessionID = event["sessionID"]
    userID = event["userID"]
    #query = event["queryStringParameters"]["q"]
    #sessionID = "1232321"
    #TIME
    start_time = time.time()

    rand = random.randint(0, 1)
    
    if rand % 2 == 0:
        openai.api_key = "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"
        os.environ["OPENAI_API_KEY"] = "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"
    else:
        openai.api_key = "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"
        os.environ["OPENAI_API_KEY"] = "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"
        

    PINECONE_CACHE_KEY='e2376134-487b-4008-8ddb-60bfd28e489b'
    PINECONE_CACHE_ENV='us-west1-gcp-free'

    pinecone.init(
        api_key=PINECONE_CACHE_KEY,
        environment=PINECONE_CACHE_ENV
    )


    index_cache = pinecone.Index("lynk-summary-cache")


    questions = []
    PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
    PINECONE_ENV = 'us-central1-gcp'


    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)
    xq = openai_embed.embed_query(query)
    

    xc = index_cache.query(xq, top_k=10, include_metadata=True)
    written_dynamo_db = False
    try: 
        for result in xc['matches']:
            if result['score'] < .97:
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
            end_time = time.time()
            time_diff = end_time - start_time
            print("GET CACHE PINECONE: {:.3f} seconds".format(time_diff))
    
            return {
                'statusCode': 200,
                'headers': {
                                   "Content-Type" : "application/json",
                "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods" : "OPTIONS,POST",
                "Access-Control-Allow-Credentials" : 'true',
                "Access-Control-Allow-Origin" : "*",
                "X-Requested-With" : "*"
                },
                'body':  answer
            }
        if len(xc['matches']) == 0:
            str_id = f'{uuid.uuid4()}'
            record = (str_id, xq, {'text': query})
            index_cache.upsert(vectors=[record])
            written_dynamo_db = True

            print("UPDATED PINECONE")

    except Exception as e:
        print('ERROR SINCE NO INDEX')
        print(e)
        str_id = f'{uuid.uuid4()}'
        record = (str_id, xq, {'text': query})
        index_cache.upsert(vectors=[record])
        written_dynamo_db = True

        
    end_time = time.time()
    time_diff = end_time - start_time
    # print("GET CACHE PINECONE: {:.3f} seconds".format(time_diff))
    
    # query dynamo db to get all comments
    start_time = time.time()

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    index = pinecone.Index("lynk-questions")
    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)


    # query Pinecone to get all Questions
    all_comment_dfs = []
    questions = []
    question_text = {}
    xc = index.query(xq, top_k=10, include_metadata=True)
    total_tokens = 0

    for result in xc['matches']:
        if result['score'] < .9:
            break

        response = questions_table.query(
            IndexName='title-index',
            KeyConditionExpression=Key('title').eq(result['metadata']['text'])
        )

        #print(result['metadata']['text'])
        for item in response['Items']:
            #sub_id = str(item['id'] + "_" item['sub'])
            question_text[item['questionID']] = (item['title'], item['body'])
            questions.append(item['questionID'])

            row = {}
            row['id'] = uuid.uuid1().hex
            row['topics'] = item['sub']

            with topics_table.batch_writer() as batch:
                print('WRITING TO DYNAMODB')
                batch.put_item(Item=row)



    print("Questions")    
    #print(questions)
    #print(question_text)    

    #print(question_text)
    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET QUESTIONS PINECONE: {:.3f} seconds".format(time_diff))
    
    max_tokens = 10000
    total_tokens = 0
    #totalTokensOver = False
    
    def iterative_comments(df, total_tokens):
        #global total_tokens  # ensure we're updating the global total_tokens variable
    
        threads = {}
        indexed_df = df.set_index('commentId')
        
        for idx, row in df.iterrows():
            comment_id = row['commentId']
            parent_id = row['parent_id']
            body_tokens = num_tokens_from_string(row['body'], "cl100k_base")
            
            if total_tokens + body_tokens > max_tokens:
                return threads, total_tokens
            
            total_tokens += body_tokens
            
            if parent_id not in indexed_df.index:
                threads[comment_id] = {
                    'comment_id': comment_id,
                    'body': row['body'],
                    'replies': {}
                }
            else:
                parent_comment = find_parent_comment(threads, parent_id)
                if parent_comment is not None:
                    parent_comment['replies'][comment_id] = {
                        'comment_id': comment_id,
                        'body': row['body'],
                        'replies': {}
                    }
        
        return threads, total_tokens


    
    def find_parent_comment(thread, parent_id):
        # This function recursively searches the comment thread for a comment with a given id
        for comment_id, comment in thread.items():
            if comment_id == parent_id:
                return comment
            elif comment['replies']:
                found = find_parent_comment(comment['replies'], parent_id)
                if found is not None:
                    return found
        return None



    
    def transform_first_id(row):
        sub_len = -2 * len(row['sub']) - 1 
        first_id = row['firstId'][:sub_len]
        first_id = first_id.split("_")[1]
        first_id = first_id + "_" + row['sub']

        return first_id
    
     
    matches = []
    for question_id in questions:
        if total_tokens >= max_tokens:
            break
        print(question_id)
    
        question_id_t3 = "t3_" + question_id
    
        question_comments = comments_table.query(
            IndexName='parentId-index',
            KeyConditionExpression=Key('parentId').eq(question_id_t3)
        )
    
        question_comments_df = pd.DataFrame(question_comments['Items'])
        question_comments_df['parent_id'] = question_comments_df.apply(transform_first_id, axis=1)
        all_comment_dfs.append(question_comments_df)
    
        comments, total_tokens = iterative_comments(question_comments_df, total_tokens)
        
        print("tokens: ")
        print(total_tokens)
    
        match = {
            'question_id': question_id,
            'question_title': question_text.get(question_id)[0],
            'question_further_description': question_text.get(question_id)[1],
            'comments': list(comments.values())  # Convert dictionary values to a list
        }
        matches.append(match)

    
    input_json = {
        'user_question': query,
        'matching_reddit_posts': matches
    }
    
    input_json = json.dumps(input_json).replace("{", "{{").replace("}", "}}")  # Replace single curly braces with double

    print(input_json)
    
    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET COMMENTS DYNAMODB: {:.3f} seconds".format(time_diff))    

    # GPT KEY CLAIMS, SUMMARY, SUPPORTING COMMENTS -------------------------------------------------------------------------
    start_time = time.time()
    
    
    prompt = f'''You are an AI-based search engine that uses Reddit discussion threads to answer a user's question. 
    A particular user has the question: '{query}'. 
    Your job is to go through the provided Reddit threads and comments to provide a clear, actionable answer to the user's question. 
    Ignore any troll or irrelevant comments. 
    Make sure your answer is well-structured, either in intuitive sections or as a list of points for easy understanding.
    Your answer will be displayed directly to the user, so direct your answer to the user. 
    
    
    Here's the user's question and the most relevant Reddit threads relating to the question: 
    
    {input_json}
    
    '''
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "You are an AI search engine."},
            {"role": "user", "content": prompt},
        ],
      temperature=0.3,
    )
    print(response)
    #print(sd)

    
    # Extract the response content
    answer = response['choices'][0]['message']['content']
    json_output= {"summary": answer}

    print(answer)
    combined_questions_comments_df = pd.concat(all_comment_dfs)
    comments_list = combined_questions_comments_df['body'].tolist()
    

    try:
        finalResponse = {
            "comments": len(comments_list),
            "llm_resp_array" : json_output
        }
    except:
        finalResponse = {
            "comments": 0,
            "llm_resp_array" : json_output
        }

    
    print(finalResponse)

    
    end_time = time.time()
    time_diff = end_time - start_time
    print("LLM: {:.3f} seconds".format(time_diff))
    

    # write to pinecone cache in dynamoDB if not exists
    row = {}
    row['id'] = uuid.uuid1().hex
    row['question'] = query
    row['response'] = json.dumps(finalResponse)

    print(row)

    if written_dynamo_db:
        with cache_table.batch_writer() as batch:
            print('WRITING TO DYNAMODB')
            batch.put_item(Item=row)



    row['sessionID'] = sessionID
    with search_table.batch_writer() as batch:
            print('WRITING TO DYNAMODB')
            batch.put_item(Item=row)


    if userID != "null":
        row['userID'] = userID
        # Returns the current local date
        current_date = datetime.now()

        date_string = current_date.strftime("%Y-%m-%d")
        row['date'] = date_string

        with user_search_table.batch_writer() as batch:
                print('WRITING TO DYNAMODB')
                batch.put_item(Item=row)
    # return search results and headers
    return {
        'statusCode': 200,
        'headers': {
               "Content-Type" : "application/json",
                "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods" : "OPTIONS,POST",
                "Access-Control-Allow-Credentials" : 'true',
                "Access-Control-Allow-Origin" : "*",
                "X-Requested-With" : "*"
        },
        'body':  json.dumps(finalResponse)
    }


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens