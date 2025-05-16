import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(0)

device = "cpu"

class VisionSdpaAttention(nn.Module):
    def __init__(self, dim, num_heads=16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.k_proj.weight[1:, :] = 0
            self.v_proj.weight[1:, :] = 0

    def forward(self, hidden_states, attention_mask):
        seq_length = hidden_states.shape[0]
        
        #q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        with_sdpa_fix = False
        if with_sdpa_fix:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        print(q.shape)
        print(k.shape)
        print(v.shape)
        print(attention_mask.shape)


        attn_output = F.scaled_dot_product_attention(
            q, k, v, attention_mask, dropout_p=0.0
        )
        if with_sdpa_fix:
            attn_output = attn_output.squeeze(0)
        #attn_output = attn_output.transpose(0, 1)

        print("===", attn_output.shape)
        return attn_output


num_images = 2
grid_thw = torch.zeros([num_images, 3], device=device, dtype=torch.long)
for n in range(num_images):
    grid_thw[n, 0] = 1
    grid_thw[n, 1] = 2
    grid_thw[n, 2] = 2

cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    dim=0,
    # Select dtype based on the following factors:
    #  - FA2 requires that cu_seqlens_q must have dtype int32
    #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
    # See https://github.com/huggingface/transformers/pull/34852 for more information
    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

seq_length = cu_seqlens[-1]
attention_mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.float32) + torch.finfo(torch.float16).min
for i in range(1, len(cu_seqlens)):
    attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0


print(attention_mask)

print(cu_seqlens)

num_heads = 1
hidden_size = num_heads * 16*3
model = VisionSdpaAttention(hidden_size, num_heads)
hidden_states = torch.ones([seq_length, hidden_size], device=device, dtype=torch.float32)

ref = model(hidden_states, attention_mask)
ref = ref.detach().numpy()


import openvino

ovm = openvino.convert_model(model, example_input=[hidden_states, attention_mask], verbose=True)
openvino.serialize(ovm, "ovm.xml")
cvtm = openvino.compile_model(ovm, device_name="GPU")
output = cvtm([hidden_states.numpy(), attention_mask.numpy()])
res = output[0]

np.set_printoptions(linewidth=1024)
if not np.allclose(ref, res, atol=1e-3, rtol=1e-3):
    #print(ref - res)
    print(ref[0,0:2,:])
    print(res[0,0:2,:])
else:
    print(ref[0,0:2,:8])
    print(res[0,0:2,:8])
    print("PASS")