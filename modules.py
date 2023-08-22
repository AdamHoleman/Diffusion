
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, channels):
    #Follow section 3.5 of 'Attention is All You Need' - assumes number of channels is even.

    device = t.device #make sure all tensors are on the same device
    positions = (torch.arange(0,channels, 2).float()/channels).to(device)
    emb_1 = torch.sin(t/(10000**positions))
    emb_2 = torch.cos(t/(10000**positions))

    return torch.cat([emb_1, emb_2], dim = -1)


class Residual_Block(nn.Module):

    def __init__(self, in_ch, out_ch, time_emb = True):
        super().__init__()

        self.out_ch = out_ch
        self.in_ch = in_ch
        self.time_emb = time_emb

        self.conv_1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
                                   nn.GroupNorm(1, in_ch),
                                   nn.SiLU())
        self.conv_2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm_2 = nn.GroupNorm(1, out_ch)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(256, in_ch))

        #final convolution is just to allow residual connection through the block
        if self.out_ch != self.in_ch:
          self.conv_3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
          self.conv_3 = nn.Identity()

    def forward(self, x, t):
        y = self.conv_1(x)

        #add timestep embedding
        if self.time_emb:
          t = self.time_proj(t)[:,:, None, None].repeat(1,1,y.shape[-2],y.shape[-1])
          y += t

        y = self.conv_2(y)
        y = self.norm_2(y)
        x = self.conv_3(x)

        return F.gelu(y + x) #adding a non-linearity into the residual connection seems to help training dynamics



class Self_Attention(nn.Module): 
    def __init__(self, channels, size):
        super(Self_Attention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.LayerNorm([channels])
        self.mlp = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_norm = self.norm(x)
        att, _ = self.mha(x_norm, x_norm, x_norm)
        att = att + x
        att = self.mlp(att) + att
        return att.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)





class Down_Block(nn.Module):
    def __init__(self, in_ch, out_ch, im_size):
        super().__init__()

        self.conv = Residual_Block(in_ch, out_ch)
        self.dsample = nn.MaxPool2d(2)
        self.att = Self_Attention(out_ch, im_size//2)


    def forward(self, x, t):
        x = self.dsample(x)
        x = self.conv(x, t)
        x = self.att(x)

        return x





class Up_Block(nn.Module):
    def __init__(self, in_ch, out_ch, im_size):
        super().__init__()

        self.conv = Residual_Block(in_ch, out_ch)
        self.usample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att = Self_Attention(out_ch, im_size)

    def forward(self, y, t):
        y = self.usample(y)
        y = self.conv(y,t)
        y = self.att(y)

        return y





class UNET(nn.Module):
    #A mildly customizable UNET architecture, allowing the user to specify the desired number of contracting layers.
    def __init__(self, num_contracting, im_size = 32):
        super().__init__()

        assert 2**(num_contracting+1) < im_size, "Too many contracting layers for the image size"

        self.contracting_layers = nn.ModuleList()
        self.expanding_layers = nn.ModuleList()
        channel_list = [64]

        bottom_channels = (2**(num_contracting-1))*64

        #the bottleneck layers
        self.bot1 = Residual_Block(bottom_channels, 2*bottom_channels, False)
        self.bot2 = Residual_Block(2*bottom_channels, 2*bottom_channels, False)
        self.bot3 = Residual_Block(2*bottom_channels, bottom_channels, False)


        #expanding and contracting layers

        #add top layers
        self.contracting_layers.append(Residual_Block(3, 64))
        self.expanding_layers.append(Residual_Block(128, 64))

        #for each contracting layer, we add a down_block and a corresponding up_block
        for n in range(num_contracting):
            if n < num_contracting-1:
              channels = 64*(2**(n+1))
            else:
              channels = bottom_channels
            self.contracting_layers.append(Down_Block(channel_list[n], channels, im_size//(2**n)))
            channel_list.append(channels)

            self.expanding_layers.insert(0, Up_Block(2*channel_list[n+1], channel_list[n], im_size//(2**(n))))

        #bring back to rbg channels
        self.out_conv = nn.Conv2d(64, 3, kernel_size = 1)


    def forward(self, x, t):
        
        #keep track of the successive contractions of the input for the residual connections in the up blocks
        contracted_images = [x] 

        #timestep embedding
        emb = timestep_embedding(t, 256)

        for layer in self.contracting_layers:
            x = layer(x, emb)
            contracted_images.insert(0, x) #save for residual connections

        im = self.bot1(x, emb)
        im = self.bot2(im, emb)
        im = self.bot3(im, emb)

        for i, layer in enumerate(self.expanding_layers):
            im = layer(torch.cat((im, contracted_images[i]), dim = 1), emb)


        return self.out_conv(im)
        
        

        
        
        
