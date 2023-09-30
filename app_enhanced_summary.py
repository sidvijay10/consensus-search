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


client = boto3.client('kendra')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

questions_table = dynamodb.Table('6998-questions-db')
comments_table = dynamodb.Table('6998-comments-db')
cache_table = dynamodb.Table('consensus-cache')
search_table = dynamodb.Table('consensus-searches')

openai.api_key = "REDACTED"
os.environ["OPENAI_API_KEY"] = "REDACTED"

API_KEY = "REDACTED"
anthropic_client = anthropic.Client(API_KEY)


def lambda_handler(event, context):
    #print(event)
    #query = json.loads(event['body'])["query"]
    #sessionID = json.loads(event['body'])["sessionID"]
    query = event["queryStringParameters"]["q"]
    sessionID = "1232321"
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

    # CLAUDE KEY CLAIMS, SUMMARY, SUPPORTING COMMENTS -------------------------------------------------------------------------
    start_time = time.time()

    comments_str = "\n".join([f"<comment id='{i+1}'>{comment}</comment>" for i, comment in enumerate(comments)])

    prompt_Claude = f''' Here's a series of numbered Reddit comments in response to the question \"{query}\" by a user: 
    <comments>
    {comments_str}
    </comments>
    
    You are an AI language model tasked with analyzing the provided numbered Reddit comments in response to the question \"{query}\" and perform the following task:
    
    1. Create five distinct groups, or "clusters", out of the comments. Each cluster should represent a unique key claim or central argument that is common to the comments within that cluster. Each claim must answer the question \"{query}\". You must form at least 5 different clusters. Each cluster should be made up of multiple comments that share the same key claim. If a cluster contains only a single comment, do not include it in the final output. For each identified cluster, provide the overarching key claim that unites the comments in this cluster, a detailed summarization that explains the main points made in the comments of this cluster, and the comment id of the comments that belong to this cluster.
    
    Disregard any comments that are troll-like or irrelevant, focusing instead on those that provide thoughtful, meaningful responses to the question. The final output should be structured as a single XML block, which encapsulates a set of "cluster" elements. Each "cluster" element represents a cluster and must contain the elements: "cluster_number", "claim", "summary", and "comments_in_cluster". 
    
    You must return your response to the task using this XML format. Here is the expected response format:
    
    <claude_resp_array>
        <clusters>
            <cluster>
                <cluster_number>1</cluster_number>
                <claim>Shared claim in cluster 1</claim>
                <summary>Your cluster 1 summary</summary>
                <comments_in_cluster>1, 4, 11, 14</comments_in_cluster>
            </cluster>
            <cluster>
                <cluster_number>2</cluster_number>
                <claim>Shared claim in cluster 2</claim>
                <summary>Your cluster 2 summary</summary>
                <comments_in_cluster>2, 3, 15, 17</comments_in_cluster>
            </cluster>
            <!-- Additional clusters and their details go here -->
        </clusters>
    </claude_resp_array>
    
    Your output can be as long as you need it to be. Do not omit any clusters for brevity.
    
    Before you provide the XML response, do you understand the instructions? 
    '''
    starting_phrase_Claude = f"Yes, I will analyze the comments and form at least 5 clusters with shared claims to answer the question \"{query}\". Each cluster will include a summary and a list of related comment IDs. I will provide my response in the XML format without omitting any clusters for brevity."
    human_response = "Great. Now give me your XML output."
    prompt_Claude = f"{anthropic.HUMAN_PROMPT} {prompt_Claude}{anthropic.AI_PROMPT}{starting_phrase_Claude}{anthropic.HUMAN_PROMPT} {human_response}{anthropic.AI_PROMPT}"
    def LLM_completion(task_id, temp, top_k, top_p, model_name):

        response = anthropic_client.completion(prompt=prompt_Claude, 
                                               model=model_name, 
                                               stop_sequences=[anthropic.HUMAN_PROMPT],
                                               max_tokens_to_sample=2000,temperature = temp
                                               )
        llm_resp = response["completion"]
        #print("CLAUDE CLAIMS: ")
        #print(llm_resp)
        return llm_resp
    
    
    claude_output = LLM_completion(task_id = 0, temp = 0.4, top_k = -1, top_p = -1, model_name = "claude-instant-v1.1-100k") #model_name = "claude-v1")
    
    
    # Parse the XML and convert to Python dictionary
    root = ET.fromstring(claude_output)
    
    def xml_to_dict(root):
        return {root.tag: list(map(xml_to_dict, root)) if len(root) > 0 else root.text for root in root}
    
    claims_dict = xml_to_dict(root)
    
    for cluster in claims_dict["clusters"]:
        indices = map(int, cluster["comments_in_cluster"].replace(' ', '').split(','))
        cluster["comments_in_cluster"] = [comments[i-1] for i in indices]  # subtract 1 to handle 0-based indexing in Python

    #print(claims_dict)
    end_time = time.time()
    time_diff = end_time - start_time

    print("CLAUDE EXTRACT CLAIMS: {:.3f} seconds".format(time_diff))
    
    # CLAUDE COMPREHENSIVE SUMMARY: SUMMARY, AGREEMENT, DISAGREEMENT, CONSENSUS -----------------------------------------
    start_time = time.time()
    summary_prompt_Claude = f''' Here's a series of numbered Reddit comments in response to the question "{query}" by a user:
    <comments>
    {comments_str}
    </comments>
    
    You are an AI language model tasked with analyzing the provided numbered Reddit comments in response to the question "{query}" and perform the following tasks:
    
    Create an in-depth, thorough summary encapsulating various perspectives expressed in the comments.
    Highlight key arguments, detect areas of consensus and disagreement.
    Provide explanations for complex or contentious points, using examples from the comments when required.
    Identify trends and contradictions, presenting an analytical overview of the discussion beyond a simple restatement of the comments.
    Exclude any troll-like or irrelevant remarks, focusing primarily on thoughtful, substantial responses to the question. The final output should be formatted into a neatly structured XML block, capturing your comprehensive analysis.
    
    The desired response format should look like:
    
    <llm_resp>
    <summary_section>
    </summary_section>
    <areas_of_disagreement_section>
    </areas_of_disagreement_section>
    <areas_of_agreement_section>
    </areas_of_agreement_section>
    <overall_consensus_section>
    </overall_consensus_section>
    </llm_resp>

    You must return in this XML format.
    '''
    summary_prompt_Claude = f"{anthropic.HUMAN_PROMPT} {summary_prompt_Claude}{anthropic.AI_PROMPT}"
    
    def LLM_summary(temp, top_k, top_p, model_name):

        response = anthropic_client.completion(prompt=summary_prompt_Claude, 
                                               model=model_name, 
                                               stop_sequences=[anthropic.HUMAN_PROMPT],
                                               max_tokens_to_sample=2000,temperature = temp
                                               )
        llm_resp = response["completion"]
        #print("CLAUDE SUMMARY: ")
        #print(llm_resp)
        return llm_resp
    
    
    claude_summary_output = LLM_summary(temp = 0.4, top_k = -1, top_p = -1, model_name = "claude-instant-v1.1-100k") #model_name = "claude-v1")
    #print(claude_summary_output)
    
    # Parse the XML
    root = ET.fromstring(claude_summary_output)
    def xml_to_dict(element):
        return {element.tag: list(map(xml_to_dict, element)) if len(element) > 0 else element.text}
    
    summary_data = xml_to_dict(root)
    #print(summary_data)
    
    claims_dict["summary"] = "".join(summary_data['llm_resp'][0].values())
    claims_dict["areas_of_disagreement_section"] = "".join(summary_data['llm_resp'][1].values())
    claims_dict["areas_of_agreement_section"] = "".join(summary_data['llm_resp'][2].values())
    claims_dict["overall_consensus_section"] = "".join(summary_data['llm_resp'][3].values())
    
    
    end_time = time.time()
    time_diff = end_time - start_time

    print("CLAUDE EXTRACT SUMMARY: {:.3f} seconds".format(time_diff))

    response = {
        "comments": comments[:],
        "llm_resp_array" : claims_dict
    }
    print(response)
    

    # write to pinecone cache in dynamoDB if not exists
    row = {}
    row['id'] = uuid.uuid1().hex
    row['question'] = query
    row['response'] = response

    if written_dynamo_db:
        with cache_table.batch_writer() as batch:
            print('WRITING TO DYNAMODB')
            batch.put_item(Item=row)



    row['session_id'] = sessionID
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
