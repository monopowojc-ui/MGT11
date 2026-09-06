import torch
import torch.nn as nn

from torch_geometric.nn import GATConv



class GraphMAE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        embed_dim=32
    ):

        super().__init__()


        self.encoder1=GATConv(
            input_dim,
            hidden_dim,
            heads=2,
            concat=True
        )


        self.encoder2=GATConv(
            hidden_dim*2,
            embed_dim,
            heads=1,
            concat=False
        )



        self.decoder=nn.Linear(
            embed_dim,
            input_dim
        )



    def forward(
        self,
        x,
        edge_index
    ):


        h=self.encoder1(
            x,
            edge_index
        )

        h=torch.relu(h)


        z=self.encoder2(
            h,
            edge_index
        )


        x_hat=self.decoder(
            z
        )


        return z,x_hat