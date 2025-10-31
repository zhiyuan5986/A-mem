
export HF_ENDPOINT=https://hf-mirror.com
# python test_advanced.py --model llama3.2:3b \
#                         --output outputs/results_llama3.2:3b.json \
#                         --backend ollama

python test_advanced.py --model llama3.2:1b \
                        --output outputs/results_llama3.2:1b.json \
                        --backend ollama

# python test_advanced.py --model qwen2.5:3b \
#                         --output outputs/results_qwen2.5:3b.json \
#                         --backend ollama

# python test_advanced.py --model qwen2.5:1.5b \
#                         --output outputs/results_qwen2.5:1.5b.json \
#                         --backend ollama
