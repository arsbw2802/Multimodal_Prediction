import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(ConvLSTMEncoder, self).__init__()
        self.encoder = Convolutional1DEncoder(args)
        self.rnn = nn.GRU(input_size=128,
                          hidden_size=256,
                          num_layers=2,
                          bidirectional=False,
                          batch_first=True,
                          dropout=0.2)

    def forward(self, inputs):
        z = self.encoder(inputs)
        # Passing through the RNN
        r_out, _ = self.rnn(z, None)

        return r_out


class Convolutional1DEncoder(nn.Module):
    def __init__(self, args):
        super(Convolutional1DEncoder, self).__init__()
        #print("INPUT SIZE HERE")
        print(args.input_size)
    
        self.encoder = nn.Sequential(
            ConvBlock(args.input_size, 32, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect'),
            ConvBlock(32, 64, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect'),
            ConvBlock(64, 128, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect')
        )
        

    def forward(self, inputs):
        # Tranposing since the Conv1D requires
        inputs = inputs.permute(0, 2, 1)
        encoder = self.encoder(inputs)
        encoder = encoder.permute(0, 2, 1)
       

        return encoder


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=1, padding_mode='reflect', dropout_prob=0.2):
        super(ConvBlock, self).__init__()

        # 1D convolutional layer
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        conv = self.conv(inputs)
        relu = self.relu(conv)
        dropout = self.dropout(relu)

        return dropout
    
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

    
        self.args= args
    
        
        self.encoder = ConvLSTMEncoder(args)
        self.softmax = nn.Linear(256,  args.num_classes)

        
        # Initialize Dense layer weights
        nn.init.xavier_uniform_(self.softmax.weight)
        nn.init.zeros_(self.softmax.bias)



    def forward(self, inputs):
     
        
        encoder = self.encoder(inputs)
        softmax = self.softmax(encoder[:,-1,:])
        
      
        return softmax

    def load_pretrained_weights(self, args, few_shot=False):
        state_dict_path = os.path.join(args.saved_model)

        print('Loading the pre-trained weights')
        checkpoint = torch.load(state_dict_path, map_location=args.device)
       
        pretrained_checkpoint = checkpoint['model_state_dict']

        model_dict = self.state_dict()

        print(model_dict.keys(), pretrained_checkpoint.keys())
        
            
        updated_checkpoints = {}
        for k, v in pretrained_checkpoint.items():
            updated_checkpoints[ k] = v
        

                        
        # What weights are *not* copied
        missing = \
            {k: v for k, v in updated_checkpoints.items() if
            k not in model_dict}
        print("The weights from saved model not in classifier are: {}".format(
            missing.keys()))

        missing = \
            {k: v for k, v in model_dict.items() if
            k not in updated_checkpoints}
        print("The weights from classifier not in the saved model are: {}"
            .format(missing.keys()))
            
            
        

        self.load_state_dict(updated_checkpoints, False)
        


        return

    def freeze_encoder_layers(self):
        """
        To set only the softmax to be trainable
        :return: None, just setting the encoder part (or the CPC model) as
        frozen
        """
        # First setting the model to eval
        
        self.encoder.eval()
        # self.embedding.eval()
       

        # Then setting the requires_grad to False
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
            
       

        return
