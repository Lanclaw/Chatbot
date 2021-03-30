import torch
from DataProcess import *
import random
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker


def evaluate(encoder, decoder, input_lang, output_lang, source, max_length=MAX_LENGTH):
    src_tensor = sen2tensor(input_lang, source)

    enc_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    enc_hidden = None
    for i in range(src_tensor.size(0)):
        enc_output, enc_hidden = encoder(src_tensor[i], enc_hidden)
        enc_outputs[i] = enc_output[0, 0]

    dec_hidden = enc_hidden
    dec_input = torch.tensor([[SOS_token]], device=device)
    attn_mat = torch.zeros(max_length, max_length)
    dec_outputs = []
    for i in range(max_length):
        dec_output, dec_hidden, attn = decoder(dec_input, dec_hidden, enc_outputs)
        attn_mat[i] = attn[0]
        _, topi = dec_output.topk(1)
        dec_input = topi.squeeze().detach().view(1, -1)
        if dec_input == EOS_token:
            break
        dec_outputs.append(dec_input.item())

    return dec_outputs, attn_mat


def evaluateRandomly(encoder, decoder, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('src:', pair[0])
        print('real:', pair[1])
        output_words = evaluate(encoder, decoder, input_lang, output_lang, pair[0])
        output_sentence = ' '.join(id2sen(output_lang, output_words))
        print('pred:', output_sentence)
        print('')


def displayAttention(candidate, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    attention = attention.squeeze(1).cpu().detach().numpy()
    cax = ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)

    ax.set_xticklabels([''] + candidate.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def evaluateAndShowAttention(encoder, attn_decoder, input_lang, output_lang, input_sentence):
    output_words, attentions = evaluate(encoder, attn_decoder, input_lang, output_lang, input_sentence)
    output_words = id2sen(output_lang, output_words)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    displayAttention(input_sentence, output_words, attentions)