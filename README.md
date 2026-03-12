# helios
local LLM inference serving engine

To download Mistral-7B-Instruct from HuggingFace
python -c "      
>> from huggingface_hub import hf_hub_download
>> hf_hub_download(
>>     repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
>>     filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
>>     local_dir='./models'
>> )
>> "