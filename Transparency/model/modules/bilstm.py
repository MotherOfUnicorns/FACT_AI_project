"""Implementation of batch-normalized BiLSTM."""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init

class BiLSTM(nn.Module):

    """A module that runs multiple steps of BiLSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(BiLSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell_forward = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            cell_backward = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'forw_cell_{}'.format(layer),cell_forward)
            setattr(self, 'backw_cell_{}'.format(layer),cell_backward)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return [getattr(self, 'forw_cell_{}'.format(layer)),getattr(self, 'backw_cell_{}'.format(layer))]

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell_forward = self.get_cell(layer)[0]
            cell_backward = self.get_cell(layer)[1]
            cell_forward.reset_parameters()
            cell_backward.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        
        output = []
        cell_output = []

        for time in range(max_time):
            h_next, c_next = cell(input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)

            output.append(h_next)
            cell_output.append(c_next)
            hx = hx_next
        
        output = torch.stack(output, 0)
        cell_output = torch.stack(cell_output,0)

        return output, cell_output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
                hx_forward = (Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size,device=device))),
                  Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size,device=device))))
                hx_backward = (Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size,device=device))),
                  Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size,device=device))))
            else:
                hx_forward = (Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size))),
                  Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size))))
                hx_backward = (Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size))),
                  Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size))))
        # h_n = []
        # c_n = []
        output_states = []
        layer_output_forward = None
        layer_cell_output_forward = None
        layer_output_backward = None
        layer_cell_output_backward = None
        for layer in range(self.num_layers):
            cell_forward = self.get_cell(layer)[0]
            cell_backward = self.get_cell(layer)[1]
            hx_layer_forward = (hx_forward[0][layer,:,:], hx_forward[1][layer,:,:])
            hx_layer_backward = (hx_backward[0][layer,:,:], hx_backward[1][layer,:,:])
            
            if layer == 0:
                layer_output_forward, layer_cell_output_forward, (layer_h_n_forward, layer_c_n_forward) = BiLSTM._forward_rnn(
                    cell=cell_forward, input_=input_, length=length, hx=hx_layer_forward)
                input_backward = torch.flip(input_,[0])
                layer_output_backward, layer_cell_output_backward, (layer_h_n_backward, layer_c_n_backward) = BiLSTM._forward_rnn(
                    cell=cell_backward, input_=input_backward, length=length, hx=hx_layer_backward)
            else:
                raise(NotImplementedError)
                #not implemented
                layer_output, layer_cell_output, (layer_h_n, layer_c_n) = BiLSTM._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)
            
            input_ = self.dropout_layer(layer_output_forward)
            input_backward = self.dropout_layer(layer_output_backward)
        
        output_forward = layer_output_forward
        #flip hidden states of backward LSTM cell to concatenate corresponding words as in regular BiLSTM definition
        output_backward = torch.flip(layer_output_backward,[0])
        output = torch.cat((output_forward,output_backward),dim=2)
        cell_output_forward = layer_cell_output_forward
        #flip cell states of backward LSTM cell to concatenate corresponding words as in regular BiLSTM definition
        cell_output_backward = torch.flip(layer_cell_output_backward,[0])
        cell_output = torch.cat((cell_output_forward,cell_output_backward),dim=2)

        #getting the hidden and cell states for the last word
        layer_h_n = output[-1,:,:]
        layer_c_n = cell_output[-1,:,:]
        output_states = [(layer_h_n,layer_c_n)]

        if self.batch_first:
            output = torch.transpose(output,0,1)
            cell_output = torch.transpose(cell_output,0,1)

        return output, cell_output, output_states
