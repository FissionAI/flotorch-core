import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def estimate_effective_kb_tokens(configuration, num_tokens_kb_data):
    chunking_strategy = configuration["chunking_strategy"].lower()
    if chunking_strategy == 'fixed':
        chunk_size = configuration["chunk_size"]
        chunk_overlap = configuration["chunk_overlap"]

        eff_chunk_size = chunk_size * (1 - chunk_overlap / 100)

    elif chunking_strategy == 'hierarchical':
        child_chunk_size = configuration["hierarchical_child_chunk_size"]
        child_chunk_overlap = configuration["hierarchical_chunk_overlap_percentage"]

        eff_chunk_size = child_chunk_size * (1 - child_chunk_overlap / 100)

    num_chunks = num_tokens_kb_data / float(eff_chunk_size)
    effective_num_tokens = num_chunks * float(eff_chunk_size)

    return effective_num_tokens

def estimate_embedding_model_bedrock_price(file_path, configuration, effective_kb_tokens):
    try:
        df = file_path.copy()
        # return df
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        return None
    region = configuration["region"]

    embed_model = configuration["embedding_model"]

    embed_model_price = df[(df['model'] == embed_model) & (df['Region'] == region)]['input_price']
    if embed_model_price.empty:
        logger.warning("Returning price as Zero, as model is not present in Sheet")
        return 0
    else:
        embed_model_price = float(embed_model_price.values[0])  # this price is in 1000s of tokens not millions
        embed_price = embed_model_price * effective_kb_tokens / 1000000
        return embed_price


def estimate_retrieval_model_bedrock_price(file_path, configuration, avg_prompt_length,
                                           num_prompts):
    try:
        df = file_path.copy()
        # return df
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        return None
    region = configuration["region"]
    
    chunking_strategy = configuration["chunking_strategy"].lower()
    if chunking_strategy == 'fixed':
        chunk_size = configuration["chunk_size"]

    elif chunking_strategy == 'hierarchical':
        chunk_size = configuration["hierarchical_parent_chunk_size"]

    gen_model = configuration["retrieval_model"]
    n_shot_prompts = configuration["n_shot_prompts"]
    k = configuration["knn_num"]

    gen_model_price = df[(df['model'] == gen_model) & (df['Region'] == region)]['input_price']
    if gen_model_price.empty:
        logger.warning("Returning price as Zero, as model is not present in Sheet")
        return 0
    else:
        gen_model_price = float(gen_model_price.values[0])  # this price is in millions of tokens
        gen_model_out_price = df[(df['model'] == gen_model) & (df['Region'] == region)]['output_price']
        gen_model_out_price = float(gen_model_out_price.values[0])  # this price is in millions of tokens
        context_len = k * chunk_size
        prompt_len = (n_shot_prompts + 1) * avg_prompt_length
        total_input_tokens = (context_len + prompt_len) * num_prompts
        retrieval_input_price = gen_model_price * float(total_input_tokens) / 1000000
        total_output_tokens = avg_prompt_length
        retrieval_output_price = gen_model_out_price * float(total_output_tokens) / 1000000
        retrieval_price = retrieval_input_price + retrieval_output_price
        return retrieval_price



def estimate_fargate_price(total_time, vpc=8, mem=16):
    #Fargate pricing
    fargate_cpu = 0.04048 
    fargate_mem = 0.004445
    fargate_cpu_price = fargate_cpu * vpc
    fargate_mem_price = fargate_mem * mem

    fargate_price = (fargate_cpu_price + fargate_mem_price) * total_time / 3600
    return fargate_price
    
def estimate_opensearch_price(time):
    opensearch_instance_cost_per_hour = .711 # r7g.2xlarge.search
    num_instance = 3
    instance_price =  opensearch_instance_cost_per_hour * num_instance #per hour
    ebs_volume_size = 2
    ebs_volume_price = .122
    ebs_price = ebs_volume_price * ebs_volume_size * num_instance /30 /24 #3 instances for 2GB per hour
    iops_price_per_hour = 13000  # instances per hour for 16000 IOPS (3000 free)
    iops_price = iops_price_per_hour * .008 * num_instance /30/ 24 #3 instances
    
    total_price = ((instance_price + ebs_price + iops_price) * time / 3600)  #price per experiment

    return total_price

def estimate_sagemaker_price(time, number_of_instances = 1):
    sagemaker_price = 1.210 #per hour ml.g5.2xlarge per model
    overall_cost = sagemaker_price * number_of_instances * (time / 60)
    
    return overall_cost

def estimate_times(no_of_kb_tokens, num_prompts, configuration):
    # For every 50,000 tokens of kb data and 50 prompts of gt data, estimated time in mins
    estimated_time = {
        "indexing": {"sagemaker" : 2, "bedrock" : 1},
        "retrieval": {"sagemaker": 3, "bedrock": 2},
        "eval": {"sagemaker": 7, "bedrock": 6}
    }
    indexing_service = configuration['embedding_service']
    retrieval_service = configuration['retrieval_service']
    
    indexing_time = (no_of_kb_tokens/ 50000) * estimated_time['indexing'][indexing_service]
    retrieval_time = (num_prompts / 50) * estimated_time['retrieval'][retrieval_service]
    eval_time = (num_prompts / 50) * estimated_time['eval'][retrieval_service]

    return indexing_time, retrieval_time, eval_time