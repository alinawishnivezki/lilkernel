import torch
import triton
import triton.language as tl
from transformers import AutoTokenizer, BartForConditionalGeneration
from torch.autograd import profiler

# load model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to('cuda')

@triton.jit
def softmax_kernel(input_ptr, output_ptr, num_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * num_cols

    # Load row
    input = tl.load(input_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < num_cols, other=-float('inf'))

    # Stability adjustment by subtracting max
    row_max = tl.max(input, axis=0)
    input = input - row_max

    # Exp and sum
    exp_input = tl.exp(input)
    row_sum = tl.sum(exp_input, axis=0)

    # Normalize by sum
    softmax_output = exp_input / row_sum

    # Write back output
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), softmax_output, mask=tl.arange(0, BLOCK_SIZE) < num_cols)

# custom function for softmax application via Triton
def apply_softmax_tensors(input_tensor):
    batch_size, num_cols = input_tensor.size()[:2]
    BLOCK_SIZE = 128  # You can adjust based on power of 2 and sequence length

    output_tensor = torch.empty_like(input_tensor, device='cuda')
    softmax_kernel[(batch_size,)](
        input_tensor, output_tensor, num_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return output_tensor

# custom attention mechanism with Triton
class TritonAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score_vector = torch.nn.Parameter(torch.rand(embed_dim))

    def forward(self, x):
        scores = torch.matmul(x, self.score_vector)
        return scores

def getsummaries2(text, max_length=130, min_length=30, length_penalty=2.0, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True, max_length=512).to('cuda')
    
    encoder_hidden_states = model.model.encoder(inputs, return_dict=True).last_hidden_state

    attn_scores = apply_softmax_tensors(encoder_hidden_states.view(-1, encoder_hidden_states.size(-1)))

    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# profiling code
with profiler.profile(use_cuda=True, with_stack=True, profile_memory=True) as prof:
    getsummaries2("This is a sample support ticket text for summarization.")

]print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("profile_trace.json")  
]new_df = df.iloc[:10].copy()
with profiler.profile(use_cuda=True, profile_memory=True) as prof_df:
    new_df['summary'] = new_df['DESCRIPTION'].apply(getsummaries2)

print(prof_df.key_averages().table(sort_by="cuda_time_total", row_limit=10))
