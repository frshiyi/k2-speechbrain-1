
#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

from snowfall.training.ctc_graph import build_ctc_topo2
from speechbrain.pretrained import EncoderDecoderASR

import k2
import torch

import os
from snowfall.common import find_first_disambig_symbol
from snowfall.decoding.graph import compile_HLG
from snowfall.common import get_texts

def load_model():
    model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        #run_opts={'device': 'cuda:3'},
    )
    return model


@torch.no_grad()
def main():
    model = load_model()

    device = model.device

    # See https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech/blob/main/example.wav
    sound_file = './example.wav'

    wav = model.load_audio(sound_file)
    # wav is a 1-d tensor, e.g., [52173]
    #wav0 = wav[:-26100]
    #print("size of wav0:", wav0.size())
    #wav1 = wav[:-13200]
    #print("size of wav1:", wav1.size())
    
    #wavs = [wav,wav0,wav1] 
    
    #multiwav = torch.nn.utils.rnn.pad_sequence([wav,wav0,wav1],batch_first=True)
    #print("size of multiwav:", multiwav.size())
    
    #wavs = multiwav.float().to(device)
    #wavs = wav1.unsqueeze(0).float().to(device)
    # wavs is a 2-d tensor, e.g., [1, 52173]

    batchnum = 2
    lens = []
    wavs_ = []
    for i in range(batchnum):
        wavs_.append(wav)
        lens.append([1.0])#wavs[i].size())
    print("len list:", lens)

    wav_lens = torch.tensor(lens)
    wav_lens = wav_lens.to(device)
    
    multiwav = torch.nn.utils.rnn.pad_sequence(wavs_,batch_first=True) 
    print("size of multiwav:", multiwav.size()) 
    wavs = multiwav.float().to(device)  
    
    encoder_out = model.modules.encoder(wavs, wav_lens)
    print("size of encoder_out:", encoder_out.shape)
    # encoder_out.shape [N, T, C], e.g., [1, 82, 768]

    logits = model.hparams.ctc_lin(encoder_out)
    # logits.shape [N, T, C], e.g., [1, 82, 5000]

    log_probs = model.hparams.log_softmax(logits)
    print("size of log_probs:", log_probs.shape)
    # log_probs.shape [N, T, C], e.g., [1, 82, 5000]

    vocab_size = model.tokenizer.vocab_size()

    ctc_topo = build_ctc_topo2(list(range(vocab_size)))

    ctc_topo = k2.create_fsa_vec([ctc_topo]).to(device)


    print("size 1 of log_probs:", log_probs.size(1))
    #supervision_segments = torch.tensor([
    #    [0, 0, log_probs.size(1)],
    #    [0, 0, 41],#log_probs.size(1)],
    #    [0, 0, 61]#log_probs.size(1)]
    #    ],dtype=torch.int32)
    supervisions = []
    for i in range(batchnum):
        supervisions.append([0, 0, log_probs.size(1)])
    print("supervisions list:", supervisions)
    supervision_segments = torch.tensor(supervisions, dtype = torch.int32)



    dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

    print("-----------------------ctc_topo result---------------------------")

    lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec, 20.0, 8, 30,
                                         10000)
    print(lattices.shape)

    best_path = k2.shortest_path(lattices, True)
    print("best_path:", best_path.shape)

    i = 0
    best_path[0].draw('CTCbest_path_{}.svg'.format(str(i)), title='best_path')

    for i in range(wav_lens.size(0)):

        aux_labels = best_path[i].aux_labels
        aux_labels = aux_labels[aux_labels.nonzero().squeeze()]
        # The last entry is -1, so remove it
        aux_labels = aux_labels[:-1]
        hyp = model.tokenizer.decode(aux_labels.tolist())
        print(hyp)

    print("-----------------------HLG result---------------------------")

    # load L, G, symbol_table
    symbol_table = k2.SymbolTable.from_file('../data/lang_nosp/words.txt')
    phone_symbol_table = k2.SymbolTable.from_file('../data/lang_nosp/phones.txt')

    if not os.path.exists('./HLG.pt'):

        # logging.debug("Loading L_disambig.fst.txt")
        with open('../data/lang_nosp/L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        # logging.debug("Loading G.fst.txt")
        with open('../data/lang_nosp/G.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        HLG = compile_HLG(L=L,
                          G=G,
                          H=ctc_topo,
                          labels_disambig_id_start=first_phone_disambig_id,
                          aux_labels_disambig_id_start=first_word_disambig_id)
        torch.save(HLG.as_dict(), './HLG.pt')

        # lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec, 20.0, 8, 30,
        #                                     10000)
        lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, 8, 30,
                                             10000)
    else:
        print("Loading pre-compiled HLG")
        d = torch.load('./HLG.pt')
        HLG = k2.Fsa.from_dict(d)
        HLG = HLG.to(device)
        HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)  # ??
        HLG.requires_grad_(False)  # ??

        lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, 8, 30,
                                             10000)

    best_paths = k2.shortest_path(lattices, use_double_scores=True)

    i = 0
    best_path[0].draw('HLGbest_path_{}.svg'.format(str(i)), title='best_path')


    # assert best_paths.shape[0] == len(texts)
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    hyps = get_texts(best_paths, indices)
    # assert len(hyps) == len(texts)
    print(hyps)

    for i in range(batchnum):
        hyp_words = [symbol_table.get(x) for x in hyps[i]]  # hyp_words = [symbol_table.get(x) for x in hyps[i]]
        print(hyp_words)



if __name__ == '__main__':
    main()


























