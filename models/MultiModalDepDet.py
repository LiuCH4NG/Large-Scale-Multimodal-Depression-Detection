"""
-*- coding: utf-8 -*-
@Author     :   Md Rezwanul Haque
"""
import torch
import torch.nn as nn
from .base import BaseNet
from .Generate_Audio_Model import AudioTransformerModel
from .Generate_Visual_Model import GenerateVisualModel
from .transformer_timm import AttentionBlock, Attention

class MultiModalDepDet(BaseNet):

    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, 
                 mm_output_sizes=[256,64], 
                 fusion='ia', num_heads=4):
        super().__init__()

        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        
        self.audio_model = AudioTransformerModel(input_tdim=audio_input_size, 
                                                    label_dim=2, audioset_pretrain=True)
        
        self.visual_model = GenerateVisualModel(temporal_layers=6, number_class=2)

        ## audio conv
        self.conv1d_block_audio  = self.conv1d_block(in_channels=768, out_channels=512)
        self.conv1d_block_audio_1 = self.conv1d_block(512, 256)
        self.conv1d_block_audio_2 = self.conv1d_block(256, 128)
        self.conv1d_block_audio_3 = self.conv1d_block(128, 128)

        ## video conv
        self.conv1d_block_visual = self.conv1d_block(in_channels=768, out_channels=512)
        self.conv1d_block_visual_1 = self.conv1d_block(512, 256)
        self.conv1d_block_visual_2 = self.conv1d_block(256, 128)
        self.conv1d_block_visual_3 = self.conv1d_block(128, 128)
        
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output = nn.Linear(mm_output_sizes[-1], 1) ## DepMamba = 2, MultiModalDepDet = 1
        self.m = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)

        ## 
        e_dim               =   128
        input_dim_video     =   128
        input_dim_audio     =   128
        self.fusion         =   fusion

        if self.fusion in ['lt', 'it']:
            if self.fusion  == 'lt':
                self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
                self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
            elif self.fusion == 'it':
                # input_dim_video = input_dim_video // 2
                self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
                self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)   
        
        elif self.fusion in ['ia']:
            # input_dim_video = input_dim_video // 2
            self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
            self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)

        ## dropout
        self.fusion_dropout = nn.Dropout(p=0.5)
        self.audio_dropout  = nn.Dropout(p=0.5)
        self.visual_dropout = nn.Dropout(p=0.5)

    @staticmethod
    def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        """
        Creates a 1D convolutional block with optional padding.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (str): Padding method ('same' or 'valid').

        Returns:
            nn.Sequential: A sequential container of layers.
        """
        if padding == 'same':
            pad = kernel_size // 2  # Calculate padding for 'same' padding
        elif padding == 'valid':
            pad = 0
        else:
            raise ValueError("Padding must be 'same' or 'valid'")
        
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
    
    def forward_audio_conv1d(self, x):
        x = self.conv1d_block_audio_1(x)
        x = self.conv1d_block_audio_2(x)
        return x 
    
    def forward_visual_conv1d(self, x):
        x = self.conv1d_block_visual_1(x)
        x = self.conv1d_block_visual_2(x)
        return x 
    
    def late_transformer_fusion(self, xa, xv):
        xa = self.forward_audio_conv1d(xa)
        proj_x_a = xa ## torch.Size([8, 128, 143])

        xv = self.forward_visual_conv1d(xv)
        proj_x_v = xv # torch.Size([8, 128, 126])
       
        ### Late Transformer
        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self.av(proj_x_v, proj_x_a) # torch.Size([8, 143, 128])
        h_va = self.va(proj_x_a, proj_x_v) # torch.Size([8, 126, 128])

        return h_av, h_va
    
    def intermediate_transformer_fusion(self, xa, xv):
        xa = self.forward_audio_conv1d(xa)
        proj_x_a = xa ## torch.Size([8, 128, 143])

        xv = self.forward_visual_conv1d(xv)
        proj_x_v = xv # torch.Size([8, 128, 126])
        
        ### Intermidiate Transformer
        proj_x_a = xa.permute(0, 2, 1)
        proj_x_v = xv.permute(0, 2, 1)
        h_av = self.av1(proj_x_v, proj_x_a) # torch.Size([8, 143, 128])
        h_va = self.va1(proj_x_a, proj_x_v) # torch.Size([8, 126, 128])

        h_av = h_av.permute(0,2,1)
        h_va = h_va.permute(0,2,1)

        xa = h_av + xa  # torch.Size([8, 128, 143])
        xv = h_va + xv  # torch.Size([8, 128, 126])

        xa = self.conv1d_block_audio_3(xa) # torch.Size([8, 128, 142])
        xv = self.conv1d_block_visual_3(xv) # torch.Size([8, 128, 125])

        h_av = xa.permute(0,2,1) # torch.Size([8, 142, 128])
        h_va = xv.permute(0,2,1) # torch.Size([8, 125, 128])

        return h_av, h_va

    def intermediate_attention_fusion(self, xa, xv):
        xa = self.forward_audio_conv1d(xa)
        proj_x_a = xa ## torch.Size([8, 128, 143])

        xv = self.forward_visual_conv1d(xv)
        proj_x_v = xv # torch.Size([8, 128, 126])

        ### Intermidiate Attention 
        proj_x_a = xa.permute(0, 2, 1)
        proj_x_v = xv.permute(0, 2, 1)
        _, h_av = self.av1(proj_x_v, proj_x_a) ## torch.Size([8, 8, 143, 126])
        _, h_va = self.va1(proj_x_a, proj_x_v) ## torch.Size([8, 8, 126, 143])

        if h_av.size(1) > 1: #if more than 1 head, take average
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)
        h_av = h_av.sum([-2]) ## torch.Size([8, 1, 126]

        if h_va.size(1) > 1: #if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)
        h_va = h_va.sum([-2]) ## torch.Size([8, 1, 143])
 
        xa = h_va * xa  # torch.Size([8, 128, 143])
        xv = h_av * xv  # torch.Size([8, 128, 126])

        xa = self.conv1d_block_audio_3(xa) # torch.Size([8, 128, 142])
        xv = self.conv1d_block_visual_3(xv) # torch.Size([8, 128, 125])

        h_av = xa.permute(0,2,1) # torch.Size([8, 142, 128])
        h_va = xv.permute(0,2,1) # torch.Size([8, 125, 128])

        return h_av, h_va


    def feature_extractor(self, x, padding_mask=None):
        xa = x[:, :, 136:] ## xa: torch.Size([8, ~3404, 25])
        xv = x[:, :, :136] ## xv: torch.Size([8, ~3404, 136])
        
        xa = self.conv_audio(xa.permute(0,2,1)).permute(0,2,1) ## torch.Size([8, ~1570, 256])
        xv = self.conv_video(xv.permute(0,2,1)).permute(0,2,1) ## torch.Size([8, ~1570, 256])
       
        xa = self.audio_model.forward(xa) # torch.Size([8, 146, 768])
        xv = self.visual_model.forward_visual(xv) # torch.Size([8, 129, 768])

        ## dropout
        xa = self.audio_dropout(xa)
        xv = self.visual_dropout(xv)

        ## `Conv1D block`
        xa = xa.permute(0, 2, 1)  # Shape: [8, 768, 146]
        xa = self.conv1d_block_audio(xa) # torch.Size([8, 512, 145])
        
        xv = xv.permute(0, 2, 1)  # Shape: [8, 768, 129]
        xv = self.conv1d_block_visual(xv)  # torch.Size([8, 512, 128])

        ##-------------------------------------------------------------------
        if self.fusion == 'lt':
            h_av, h_va = self.late_transformer_fusion(xa, xv)

        elif self.fusion == 'it':
            h_av, h_va = self.intermediate_transformer_fusion(xa, xv)

        elif self.fusion == 'ia':
            h_av, h_va = self.intermediate_attention_fusion(xa, xv)
        ##-------------------------------------------------------------------

        audio_pooled = h_av.mean([1]) # torch.Size([8, 128]) # mean accross temporal dimension
        video_pooled = h_va.mean([1]) # torch.Size([8, 128])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)  # torch.Size([8, 256])

        x = self.fusion_dropout(x)

        return x

    def classifier(self, x):
        return self.output(x)
