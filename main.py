import DataProcess
import model
import torch
import random
import time
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
SOS_token = 0
EOS_token = 1


def train_sen(src_sen, tgt_sen, encoder, decoder, enc_optim, dec_optim, criterion, max_length=10):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    src_len = src_sen.size(0)
    tgt_len = tgt_sen.size(0)
    enc_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    enc_hidden = None

    for i in range(src_len):
        enc_output, enc_hidden = encoder(src_sen[i], enc_hidden)
        enc_outputs[i] = enc_output[0, 0]

    loss = 0
    dec_input = torch.tensor([[SOS_token]], device=device)
    dec_hidden = enc_hidden

    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True if random.random() > teacher_forcing_ratio else False
    if use_teacher_forcing:
        for i in range(tgt_len):
            dec_output, dec_hidden = decoder(dec_input, dec_hidden)
            loss += criterion(dec_output, tgt_sen[i])
            dec_input = tgt_sen[i]

    else:
        for i in range(tgt_len):
            dec_output, dec_hidden = decoder(dec_input, dec_hidden)
            loss += criterion(dec_output, tgt_sen[i])
            _, topi = dec_output.topk(1)
            dec_input = topi.squeeze().detach()

            if dec_input.item() == EOS_token:
                break


    loss.backward()
    enc_optim.step()
    dec_optim.step()

    return loss.item() / tgt_len


def train(pairs, encoder, decoder, input_lang, output_lang, n_iters=10000, lr=0.01):

    train_pairs = [DataProcess.pair2tensor(input_lang, output_lang, random.choice(pairs)) for _ in range(n_iters)]
    enc_optim = torch.optim.SGD(encoder.parameters(), lr=lr)
    dec_optim = torch.optim.SGD(decoder.parameters(), lr=lr)
    critirion = torch.nn.NLLLoss()

    for i in range(n_iters):
        train_pair = train_pairs[i]
        input_tensor = train_pair[0]
        output_tensor = train_pair[1]
        loss = train_sen(input_tensor, output_tensor, encoder, decoder, enc_optim, dec_optim, critirion)

        if i % 100 == 0:
            print('iter:', i, 'loss:', loss)


def asMinutes(s):
    m = math.floor(s / 60)
    s = s - m * 60
    return '%dmin %dsec' % (m, s)




if __name__ == '__main__':
    input_lang, output_lang, pairs = DataProcess.readLangs('human', 'machine')
    encoder = model.EncoderRNN(input_lang.n_words, 256)
    decoder = model.DecoderRNN(256, output_lang.n_words)
    train(pairs, encoder, decoder, input_lang, output_lang)
